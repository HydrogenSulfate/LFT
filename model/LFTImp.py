import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import utils
import math


class Implicit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_ncolors,
                 hidden_list: list = [320, 320],
                 angRes=5,
                 scale_factor: int = 2,
                 mode: str = 'bilinear',
                 feat_unfold=False,
                 local_ensemble=False,
                 cell_decode=False):

        super(Implicit, self).__init__()
        self.angRes = angRes
        self.scale_factor = scale_factor
        self.mode = mode
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        layers = []
        lastv = in_channels
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU(inplace=True))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_ncolors * (angRes ** 2)))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._linear_init_fn)

    def _linear_init_fn(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=math.sqrt(5.0))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, feat_lr, coord):
        """[summary]

        Args:
            feat_lr (torch.Tensor): [B, C, AH, AW]
            coord (torch.Tensor): [B,H'W',2]
            cell (torch.Tensor): [B, H'W', 2]

        Returns:
            torch.Tensor: [B, C, AH', AW']
        """
        B, C, UH, VW = feat_lr.shape
        H_lr = UH // self.angRes
        W_lr = VW // self.angRes
        U, V = self.angRes, self.angRes
        H_hr = H_lr * self.scale_factor
        W_hr = W_lr * self.scale_factor

        feat_hr = interpolate(feat_lr, self.angRes, self.scale_factor, self.mode)  # [B,C,UH',VW']
        feat_hr = rearrange(feat_hr, 'b c (a1 h) (a2 w) -> b  h w (a1 a2 c)', a1=U, a2=V)  # [B,H',W',UVC]

        # 准备[-1,1]的低分辨率空间坐标
        coord_spa_lr = utils.make_coord([H_lr, W_lr], flatten=False).cuda()\
            .permute(2, 0, 1)\
            .unsqueeze(0).expand(B, 2, H_lr, W_lr)  # [B,2,H,W]

        # 计算空间坐标信息
        eps_shift = 1e-6
        coord_ = coord.clone()  # [-1,1]的高分辨率坐标
        coord_[:, :, 0] += eps_shift
        coord_[:, :, 1] += eps_shift
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)  # [B,hw,2]
        q_coord = F.grid_sample(
            coord_spa_lr, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)  # [b,2,h,w]->[B,1,h'w',2(yx)]采样->[b,2,1,h'w']->[b,2,h'w']->[b,h'w',2]

        rel_coord = coord - q_coord  # 高分辨率坐标减去采样出的大尺寸下的低分辨率坐标，得到坐标差，用于下面的双线性插值
        rel_coord[:, :, 0] *= H_lr
        rel_coord[:, :, 1] *= W_lr

        rel_coord = rearrange(rel_coord, 'b (h w) c -> b h w c', h=H_hr, w=W_hr)  # [B,H',W',2]

        inp = torch.cat([feat_hr, rel_coord], dim=-1)  # [B,H',W',C+2]

        coord_ang = utils.make_coord([U, V], ranges=[[0, U], [0, V]], flatten=False).unsqueeze(0)  # [1,U,V,2]
        coord_ang = rearrange(coord_ang, 'b u v c -> b 1 1 (u v c)', u=U, v=V)
        coord_ang = coord_ang.expand([B, H_hr, W_hr, U*V*2]).cuda()
        inp = torch.cat([inp, coord_ang], dim=-1)  # [B,H',W',U*V*C+2+U*V*2]
        out = self.mlp(inp)  # [B,H',W',a1*a2*3]
        out = rearrange(out, 'b h w (a1 a2 c) -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes, c=3)
        return out

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = args.channels
        self.channels = channels
        self.angRes = args.angRes
        self.factor = args.scale_factor
        layer_num = 4

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(3,
                      channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      dilation=1,
                      bias=False), )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels,
                      channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      dilation=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels,
                      channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      dilation=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels,
                      channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      dilation=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = self.make_layer(layer_num=layer_num)

        ####################### UP Sampling ###########################
        # self.upsampling = nn.Sequential(
        #     nn.Conv2d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
        #     nn.PixelShuffle(self.factor),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        # )
        self.upsample = Implicit(
            in_channels=channels*(self.angRes**2) + 2 + 2*(self.angRes**2),
            out_ncolors=3,
            hidden_list=[320, 320],
            angRes=self.angRes,
            scale_factor=self.factor,
            mode='bilinear',
            feat_unfold=False,
            local_ensemble=False,
            cell_decode=False
        )

    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(
                AltFilter(self.angRes, self.channels, self.MHSA_params))
        return nn.Sequential(*layers)

    def forward(self, lr, coord):
        # Bicubic
        # lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')
        # [B(atch), 1, A(ngRes)*h(eight)*S(cale), A(ngRes)*w(idth)*S(cale)]

        # reshape for LFT
        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        # [B, C(hannels), A^2, h, w]

        for m in self.modules():
            m.h = lr.size(-2)
            m.w = lr.size(-1)

        # Initial Convolution
        buffer = self.conv_init0(lr)
        buffer = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]

        # Position Encoding
        spa_position = self.pos_encoding(buffer,
                                         dim=[3, 4],
                                         token_dim=self.channels)
        ang_position = self.pos_encoding(buffer,
                                         dim=[2],
                                         token_dim=self.channels)
        for m in self.modules():
            m.spa_position = spa_position
            m.ang_position = ang_position

        # Alternate AngTrans & SpaTrans
        buffer = self.altblock(buffer) + buffer

        # Up-Sampling using Implicit representation.
        buffer = rearrange(buffer, 'b c (a1 a2) h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        out = self.upsample(buffer, coord)
        return out


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size(
        )) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0,
                                  self.token_dim - 1,
                                  self.token_dim,
                                  dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature**grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(
                0, length - 1, length, dtype=torch.float32).view(-1, 1) /
                       grid_dim).to(x.device)
            pos_dim = torch.cat(
                [pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)


class SpaTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(SpaTrans, self).__init__()
        self.angRes = angRes
        self.kernel_field = 3
        self.kernel_search = 5
        self.spa_dim = channels * 2
        self.MLP = nn.Linear(channels * self.kernel_field**2,
                             self.spa_dim,
                             bias=False)

        self.norm = nn.LayerNorm(self.spa_dim)
        self.attention = nn.MultiheadAttention(self.spa_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.spa_dim),
            nn.Linear(self.spa_dim, self.spa_dim * 2, bias=False),
            nn.ReLU(True), nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.spa_dim * 2, self.spa_dim, bias=False),
            nn.Dropout(MHSA_params['dropout']))
        self.linear = nn.Sequential(
            nn.Conv3d(self.spa_dim,
                      channels,
                      kernel_size=(1, 1, 1),
                      padding=(0, 0, 0),
                      dilation=1,
                      bias=False))

    @staticmethod
    def gen_mask(h: int, w: int, k: int):
        atten_mask = torch.zeros([h, w, h, w])
        k_left = k // 2
        k_right = k - k_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_left):min(h, i + k_right),
                     max(0, j - k_left):min(h, j + k_right)] = 1
                atten_mask[i, j, :, :] = temp

        atten_mask = rearrange(atten_mask, 'a b c d -> (a b) (c d)')
        atten_mask = atten_mask.float().masked_fill(atten_mask == 0, float('-inf')).\
            masked_fill(atten_mask == 1, float(0.0))

        return atten_mask

    def SAI2Token(self, buffer):
        buffer = rearrange(buffer, 'b c a h w -> (b a) c h w')
        # local feature embedding
        spa_token = F.unfold(buffer,
                             kernel_size=self.kernel_field,
                             padding=self.kernel_field // 2).permute(2, 0, 1)
        spa_token = self.MLP(spa_token)
        return spa_token

    def Token2SAI(self, buffer_token_spa):
        buffer = rearrange(buffer_token_spa,
                           '(h w) (b a) c -> b c a h w',
                           h=self.h,
                           w=self.w,
                           a=self.angRes**2)
        buffer = self.linear(buffer)
        return buffer

    def forward(self, buffer):
        atten_mask = self.gen_mask(self.h, self.w,
                                   self.kernel_search).to(buffer.device)

        spa_token = self.SAI2Token(buffer)
        spa_PE = self.SAI2Token(self.spa_position)
        spa_token_norm = self.norm(spa_token + spa_PE)

        spa_token = self.attention(query=spa_token_norm,
                                   key=spa_token_norm,
                                   value=spa_token,
                                   need_weights=False,
                                   attn_mask=atten_mask)[0] + spa_token
        spa_token = self.feed_forward(spa_token) + spa_token
        buffer = self.Token2SAI(spa_token)

        return buffer


class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True), nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout']))

    @staticmethod
    def SAI2Token(buffer):
        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token,
                           '(a) (b h w) (c) -> b c a h w',
                           a=self.angRes**2,
                           h=self.h,
                           w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer


class AltFilter(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.spa_trans = SpaTrans(channels, angRes, MHSA_params)
        self.ang_trans = AngTrans(channels, angRes, MHSA_params)

    def forward(self, buffer):
        buffer = self.ang_trans(buffer)
        buffer = self.spa_trans(buffer)

        return buffer


def interpolate(x, angRes, scale_factor, mode):
    [B, C, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, C, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3,
                                  5).contiguous().view(B * angRes**2, C, h, w)
    x_upscale = F.interpolate(x_upscale,
                              scale_factor=scale_factor,
                              mode=mode,
                              align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, C, h * scale_factor,
                               w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2,
                                  5).contiguous().view(B, C, H * scale_factor,
                                                       W * scale_factor)
    # [B, 1, A*h*S, A*w*S]

    return x_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):

    pass
