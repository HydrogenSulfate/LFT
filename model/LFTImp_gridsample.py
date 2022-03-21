import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.utils import make_coord

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    #     self.apply(self._init_fn)

    # def _init_fn(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight, a=math.sqrt(5.0))
    #         if m.bias is not None:
    #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
    #             bound = 1 / math.sqrt(fan_in)
    #             nn.init.uniform_(m.bias, -bound, bound)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    #         if m.bias is not None:
    #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
    #             bound = 1 / math.sqrt(fan_in)
    #             nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.layers(x)
        return x


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = args.channels
        self.channels = channels
        self.angRes = args.angRes
        self.factor = args.scale_factor
        self.feat_unfold = args.feat_unfold
        self.local_ensemble = args.local_ensemble
        self.cell_decode = args.cell_decode
        self.local_rank = args.local_rank

        layer_num = 4

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = args.num_heads
        self.MHSA_params['dropout'] = args.dropout

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(3,
                      channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1),
                      dilation=1,
                      bias=False))
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
        self.altblock = self._make_layer(layer_num=layer_num)

        ####################### UP Sampling ###########################
        self.imnet = MLP((channels * 9 if self.feat_unfold else 1) + 2 + 2 + 2, 3, [256, 256, 256, 256])

    def _make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(
                AltFilter(self.angRes, self.channels, self.MHSA_params))
        return nn.Sequential(*layers)

    def forward_feat(self, lr):
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
        buffer = rearrange(buffer, 'b c (a1 a2) h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)

        return buffer

    def gen_feat(self, inp):
        """get LF feature in SAI format

        Args:
            inp (torch.Tensor): [B,C,UH,VW]

        """
        return self.forward_feat(inp)  # [B,C,UH,VW]

    def query_rgb(self, feat, coord, cell):
        """query_rgb

        Args:
            feat (torch.Tensor): [B,C,UH,VW]
            coord (torch.Tensor): [B,H',W',2]
            cell (torch.Tensor): [B,H',W',2]

        Returns:
            torch.Tensor: [B,C,UH,VW]
        """
        feat_batched = rearrange(feat, 'b c (u h) (v w) -> (b u v) c h w', u=self.angRes, v=self.angRes)  # [buv,c,h,w]
        if self.feat_unfold:
            feat_batched = F.unfold(feat_batched, kernel_size=3, padding=1).view(
                feat_batched.shape[0],
                feat_batched.shape[1] * 9,
                feat_batched.shape[2],
                feat_batched.shape[3]
            )  # [buv,9c,h,w]

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = (2 / (feat_batched.shape[-2])) / 2  # 一个cell的x半径
        ry = (2 / (feat_batched.shape[-1])) / 2  # 一个cell的y半径

        feat_coord_spa = make_coord([feat_batched.shape[-2], feat_batched.shape[-1]], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .expand(feat.shape[0], self.angRes, self.angRes, 2, *[feat_batched.shape[-2], feat_batched.shape[-1]])
        # [h,w,2]->[2,h,w]->[1,1,1,2,h,w]->[B,u,v,2,h,w], 得到低分辨率的[-1,1]的spatial coord
        abs_coord_ang = make_coord([self.angRes, self.angRes], flatten=False).cuda() \
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1) \
            .expand(feat.shape[0], self.angRes, self.angRes, 2, *[int(feat_batched.shape[-2]*self.factor), int(feat_batched.shape[-1]*self.factor)]) \
            .clone()
        # [u,v,2]->[1,u,v,2,1,1]->[B,u,v,2,h,w], 得到低分辨率的[-1,1]的angular coord

        feat_coord_spa = rearrange(feat_coord_spa, 'b u v c h w -> (b u v) c h w', u=self.angRes, v=self.angRes)

        abs_coord_ang = rearrange(abs_coord_ang, 'b u v c h w -> (b u v) h w c', u=self.angRes, v=self.angRes)
        abs_coord_ang *= self.angRes

        coord = coord.expand([feat_batched.shape[0], *coord.shape[1:]])  # [buv,h',w',2]
        cell = cell.expand([feat_batched.shape[0], *coord.shape[1:]])  # [buv,h',w',2]
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # [-1,1]的高分辨率坐标，顺序为(x,y)  # [buv,h',w',2]
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)  # [buv,h',w',2]
                q_feat = F.grid_sample(
                    feat_batched.float(), coord_.flip(-1),
                    mode='nearest', align_corners=False) \
                    .permute(0, 2, 3, 1)  # [buv,c,h',w']--{[buv,h',w',2(yx)]采样}->[buv,c,h',w']->[buv,h',w',c]
                q_coord = F.grid_sample(
                    feat_coord_spa.float(), coord_.flip(-1),
                    mode='nearest', align_corners=False) \
                    .permute(0, 2, 3, 1)  # [buv,2,h',w']-{[buv,h',w',2(yx)]采样}->[buv,2,h',w']->[buv,h',w',2]
                rel_coord = coord - q_coord  # 高分辨率坐标减去采样出的大尺寸下的低分辨率坐标，得到坐标差，用于下面的双线性插值
                rel_coord[:, :, :, 0] *= (feat_batched.shape[-2])  # y-h轴
                rel_coord[:, :, :, 1] *= (feat_batched.shape[-1])  # x-w轴
                rel_coord = rel_coord.expand([feat_coord_spa.shape[0], *rel_coord.shape[1:]])  # [buv,h',w',2]

                inp = torch.cat([q_feat, rel_coord, abs_coord_ang], dim=-1)  # [buv,h',w',c+2]

                if self.cell_decode:
                    rel_cell = cell.clone()  # [buv,h',w',2]
                    rel_cell[:, :, :, 0] *= (feat_batched.shape[-2])  # [buv,h',w',2]
                    rel_cell[:, :, :, 1] *= (feat_batched.shape[-1])  # [buv,h',w',2]
                    inp = torch.cat([inp, rel_cell], dim=-1)  # [buv,h',w',c+6]

                pred = self.imnet(inp)  # [buv,h',w',3]
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, :, 0] * rel_coord[:, :, :, 1])  # [buv,h',w']
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)  # [buv,h',w']

        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            """
            pred: [buv,h',w',3]
            area: [buv,h',w']
            """
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret = rearrange(ret, '(b u v) h w c -> b c (u h) (v w)', u=self.angRes, v=self.angRes)
        return ret

    def forward(self, inp, coord, cell):
        """forward process

        Args:
            inp (torch.Tensor): # [B,C,UH,VW]
            coord (torch.Tensor): # [B,H,W,2], may randomed.
            cell (torch.Tensor): # [B,H,W,2]

        Returns:
            torch.Tensor: [B,3,UH,VW]
        """
        lf_feat = self.gen_feat(inp)
        return self.query_rgb(lf_feat, coord, cell)


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
        grid_dim = self.temperature ** grid_dim
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
        self.MLP = nn.Linear(channels * self.kernel_field ** 2,
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
                           a=self.angRes ** 2)
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
                           a=self.angRes ** 2,
                           h=self.h,
                           w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)  # [L=UV, N=BHW, C=C]
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        attn_output = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)
        # attn_output, attn_output_weights = self.attention(query=ang_token_norm,
        #                            key=ang_token_norm,
        #                            value=ang_token,
        #                            need_weights=True)
        ang_token = attn_output[0] + ang_token

        # print(attn_output_weights.shape)  # 应该是[BHW,L,L]
        # attn_output_weights = rearrange(attn_output_weights, '(b h w) l1 l2 -> b h w l1 l2', b=1, h=32, w=32)
        # try:
        #     import matplotlib.pyplot as plt
        # except ImportError as e:
        #     print(e)
        # fig = plt.figure()
        # i, j = 16, 16
        # for u in range(self.angRes):
        #     for v in range(self.angRes):
        #         idx = u * self.angRes + v
        #         ax = fig.add_subplot(self.angRes, self.angRes, idx+1)
        #         # plt.subplot(self.angRes, self.angRes, idx + 1)
        #         attn_map = attn_output_weights[0, i, j, idx]  # [25]
        #         attn_map = attn_map.reshape([self.angRes, self.angRes])
        #         attn_map = attn_map.cpu().numpy()
        #         ax.axis('off')
        #         ax.imshow(attn_map, cmap=plt.cm.jet)  # 设置cmap为RGB图
        # # plt.colorbar()  # 显示色度条
        # plt.savefig('attnention_map.png', bbox_inches='tight')
        # exit(0)
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
                                  5).contiguous().view(B * angRes ** 2, C, h, w)
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


class EpiLoss(nn.Module):
    def __init__(self, args):
        super(EpiLoss, self).__init__()
        self.angRes = args.angRes
        self.reconstruction_loss = torch.nn.L1Loss()

    def gradient(self, x):
        D_dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return D_dx, D_dy

    def epi_loss(self, pred, label):
        def lf2epi(lf):
            N, an2, c, h, w = lf.shape
            an = self.angRes
            # [N, an2, h, w] -> [N*ah*h, aw, w]  &  [N*aw*w, ah, h]
            epi_h = lf.view(N, an, an, c, h, w).permute(0, 1, 4, 3, 2, 5).contiguous().view(-1, c, an, w)
            epi_v = lf.view(N, an, an, c, h, w).permute(0, 2, 5, 3, 1, 4).contiguous().view(-1, c, an, h)
            return epi_h, epi_v

        epi_h_pred, epi_v_pred = lf2epi(pred)  # [nuh,3,v,w] & [nvw,3,u,h]
        dx_h_pred, dy_h_pred = self.gradient(epi_h_pred)
        dx_v_pred, dy_v_pred = self.gradient(epi_v_pred)

        epi_h_label, epi_v_label = lf2epi(label)
        dx_h_label, dy_h_label = self.gradient(epi_h_label)
        dx_v_label, dy_v_label = self.gradient(epi_v_label)

        return self.reconstruction_loss(dx_h_pred, dx_h_label) + self.reconstruction_loss(dy_h_pred, dy_h_label) + self.reconstruction_loss(dx_v_pred, dx_v_label) + self.reconstruction_loss(dy_v_pred, dy_v_label)

    def forward(self, SR, HR):
        """calc loss.

        Args:
            SR (torch.Tensor): [B,C,UH,VW].
            HR (torch.Tensor): [B,C,UH,VW].

        Returns:
            torch.Tensor: loss, one element.
        """
        SR = rearrange(SR, 'b c (u h) (v w) -> b (u v) c h w', u=self.angRes, v=self.angRes)
        HR = rearrange(HR, 'b c (u h) (v w) -> b (u v) c h w', u=self.angRes, v=self.angRes)
        return self.epi_loss(SR, HR)


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()
        if args.epi_loss is not None and isinstance(args.epi_loss, float):
            self.criterion_epi_loss = EpiLoss(args)
            self.epi_loss = args.epi_loss
            if args.local_rank == 0:
                print("Build Epi loss and epi loss's weight = {:.2f}".format(self.epi_loss))

    def forward(self, SR, HR):
        """calc loss.

        Args:
            SR (torch.Tensor): [B,C,UH,VW].
            HR (torch.Tensor): [B,C,UH,VW].

        Returns:
            torch.Tensor: loss, one element.
        """
        loss = self.criterion_Loss(SR, HR)
        if hasattr(self, 'epi_loss'):
            loss += (self.epi_loss * self.criterion_epi_loss(SR, HR))
        return loss


def weights_init(m):

    pass
