import torch
import torch.nn as nn
# from utils.utils import MacPI2SAI, SAI2MacPI

# import torch.nn.functional as F
from einops import rearrange


class get_model(nn.Module):
    """LFATSRNet init
    Number of parameters: 43.36M
    Number of FLOPs: 406.50G(3->5, 128x128)

    Args:
        angRes_in (int, optional): angRes input. Defaults to 3.
        angRes_out (int, optional): angRes output. Defaults to 5.
    """
    def __init__(self, args: dict):
        super(get_model, self).__init__()
        self.angRes_in = args["angRes_in"]
        self.angRes_out = args["angRes_out"]
        self.encoder = Encoder(self.angRes_in, in_channels=3, depth=3)
        self.decoder = Decoder(self.angRes_in, depth=3)
        # 老版
        self.AngSRNet = AngularSR_Net(inter_plane=64,
                                      angRes_in=self.angRes_in,
                                      angRes_out=self.angRes_out,
                                      n_resblocks=1)

        # distASR模块
        # self.AngSRNet = DistgNet(n_colors=3,
        #                          angRes_in=self.angRes_in,
        #                          angRes_out=self.angRes_out,
        #                          n_group=1,
        #                          n_block=4,
        #                          channels=64)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """LFATSRNet forward

        Args:
            x1 (torch.Tensor): [b, c, ah, aw] in SAI.
            x2 (torch.Tensor): [b, c, ah, aw] in SAI.

        Returns:
            torch.Tensor: return output in SAI.
        """
        x1, m1 = sub_mean(x1)  # mean all elements in x1 to m1     x1 = x1 - m1
        x2, m2 = sub_mean(x2)  # mean all elements in x2 to m2     x2 = x2 - m2
        x1 = SAI2MacPI(x1, self.angRes_in)
        x2 = SAI2MacPI(x2, self.angRes_in)

        feats = self.encoder(x1, x2)
        out_lr = self.decoder(feats)

        out_hr = self.AngSRNet(out_lr)

        mi = (m1 + m2) / 2
        out_hr = MacPI2SAI(out_hr, self.angRes_out)
        out_hr += mi

        return out_hr


def sub_mean(x: torch.Tensor):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)  # [b,c,1,1]
    x -= mean
    return x, mean


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()
        # reflection_padding = kernel_size // 2
        reflection_padding = (kernel_size - 1) // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat,
                              out_feat,
                              stride=stride,
                              kernel_size=kernel_size,
                              bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y  # , y


class RCAB(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 kernel_size,
                 reduction,
                 norm=False,
                 act=nn.ReLU(True),
                 downscale=False,
                 return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat,
                     out_feat,
                     kernel_size,
                     stride=2 if downscale else 1,
                     norm=norm), act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction))
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat,
                                      out_feat,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out
        else:
            return out


class AngularSR_Net(nn.Module):
    def __init__(self, inter_plane, angRes_in, angRes_out, n_resblocks):
        super(AngularSR_Net, self).__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.PixelDown = PixelShuffle(1 / angRes_in)
        self.PixelUp = PixelShuffle(angRes_out)
        self.BicubicUp = nn.Upsample(size=(angRes_out, angRes_out),
                                     mode='bicubic',
                                     align_corners=True)
        self.pre_conv = nn.Conv2d(3 * angRes_in * angRes_in,
                                  inter_plane,
                                  kernel_size=3,
                                  padding=1)
        self.tail_conv = nn.Conv2d(inter_plane,
                                   3 * angRes_out * angRes_out,
                                   kernel_size=3,
                                   padding=1)

        # modules_body = [
        #     RCAB(inter_plane,
        #          inter_plane,
        #          3,
        #          reduction=16,
        #          norm=False,
        #          act=nn.ReLU(True))
        #     for _ in range(n_resblocks)
        # ]
        modules_body = [
            nn.Conv2d(3, inter_plane, kernel_size=3, stride=1, dilation=angRes_in, padding=angRes_in, bias=False)
        ]
        modules_body += [
            DistgBlock(angRes_in, inter_plane)
            for _ in range(n_resblocks)
        ]
        modules_body += [
            nn.Conv2d(inter_plane, inter_plane, kernel_size=angRes_in, stride=angRes_in, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inter_plane, inter_plane * angRes_out * angRes_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(inter_plane, 3, kernel_size=3, stride=1, dilation=angRes_out, padding=angRes_out, bias=False)
        ]

        self.detail = nn.Sequential(*modules_body)

    def forward(self, x_lr: torch.Tensor):
        """[summary]

        Args:
            x_lr (torch.Tensor): SAI with shape [b,c,ah,aw]

        Returns:
            [type]: [description]
        """
        b, c, ah, aw = x_lr.size()
        h, w = ah // self.angRes_in, aw // self.angRes_in
        # bicubic upsample for Angular
        # x_hr = x_lr.view(b, c, h, self.angRes_in, w, self.angRes_in)
        # x_hr = x_hr.permute(0, 1, 2, 4, 3, 5)  # [b,c,h,w,a,a]
        # x_hr = x_hr.reshape(b, c * h * w, self.angRes_in, self.angRes_in)  # [b,cxhxw,a,a]
        x_hr = rearrange(x_lr, 'b c (h a1) (w a2) -> b (c h w) a1 a2', a1=self.angRes_in, a2=self.angRes_in, h=h, w=w)
        x_hr = self.BicubicUp(x_hr)  # [b,cxhxw,a',a']
        # x_hr = x_hr.view(b, c, h, w, self.angRes_out, self.angRes_out)  # [b,c,h,w,a',a']
        # x_hr = x_hr.permute(0, 1, 2, 4, 3, 5)  # [b,c,h,a',w,a']
        # x_hr = x_hr.reshape(b, c, h * self.angRes_out,
        #                     w * self.angRes_out)  # [b,c,hxa',wxa']
        x_hr = rearrange(x_hr, 'b (c h w) a1 a2 -> b c (h a1) (w a2)', a1=self.angRes_out, a2=self.angRes_out, h=h, w=w)

        # detail restore
        x_res = self.detail(x_lr)
        # x_res = self.pre_conv(x_res)
        # x_res = self.detail(x_res)
        # x_res = self.tail_conv(x_res)
        # x_res = self.PixelUp(x_res)
        x_hr += x_res
        return x_hr


class DistgNet(nn.Module):
    """角度超分模块

    Args:
        n_colors (int): [description]
        angRes_in (int): [description]
        angRes_out (int): [description]
        n_group (int, optional): [description]. Defaults to 2.
        n_block (int, optional): [description]. Defaults to 2.
        channels (int, optional): [description]. Defaults to 64.
    """
    def __init__(self,
                 n_colors: int,
                 angRes_in: int,
                 angRes_out: int,
                 n_group: int = 2,
                 n_block: int = 2,
                 channels: int = 64):
        super(DistgNet, self).__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.init_conv = nn.Conv2d(n_colors, channels, kernel_size=3, stride=1, dilation=angRes_in, padding=angRes_in, bias=False)
        self.DistgGroup = CascadedDistgGroup(n_group, n_block, angRes_in, channels)
        self.UpSample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=angRes_in, stride=angRes_in, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * angRes_out * angRes_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(channels, n_colors, kernel_size=3, stride=1, dilation=angRes_out, padding=angRes_out, bias=False)
        )

    def forward(self, x):
        x = SAI2MacPI(x, self.angRes_in)
        buffer = self.init_conv(x)
        buffer = self.DistgGroup(buffer)
        out = self.UpSample(buffer)
        out = MacPI2SAI(out, self.angRes_out)
        return out


class CascadedDistgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadedDistgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DistgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.fuse = nn.Conv2d(n_group * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        for i in range(self.n_group):
            x = self.Group[i](x)
            temp.append(x)
        out = torch.cat(temp, dim=1)
        return self.fuse(out)


class DistgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DistgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DistgBlock(angRes, channels))
        self.Blocks = nn.Sequential(*Blocks)
        self.fuse = nn.Conv2d((n_block + 1) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        temp.append(x)
        for i in range(self.n_block):
            x = self.Blocks[i](x)
            temp.append(x)
        out = torch.cat(temp, dim=1)
        return self.fuse(out)


class DistgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DistgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels, channels

        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes],
                      padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            PixelShuffle1D(angRes),
        )
        self.squeezeConv = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((x, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.squeezeConv(buffer)
        y = self.SpaConv(buffer) + buffer
        return y


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tentor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y

class Decoder(nn.Module):
    def __init__(self, angRes, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = BlockShuffle(2**depth, angRes)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()  # [b,c,h,w]

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels,
                                             scale_factor, scale_factor,
                                             in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height,
                                             block_size, out_width, block_size)
        #  [b,c,h,w]->[b,c,h',bh,w',bw]
        shuffle_out = input_view.permute(0, 1, 3, 5, 2,
                                         4).contiguous()  # [b,c,bh,bw,h',w']

    return shuffle_out.view(batch_size, out_channels, out_height,
                            out_width)  # [b,c*bh*bw,h',w']


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


def block_shuffle(x, scale_factor, angRes):
    b, c, ah, aw = x.shape
    h = int(ah / angRes)
    w = int(aw / angRes)
    h_o = int(h * scale_factor)  # 2
    w_o = int(w * scale_factor)  # 2
    c_o = int(int(c / scale_factor) / scale_factor)
    if scale_factor >= 1:
        x_view = x.contiguous().view(b, c_o, scale_factor, scale_factor, h,
                                     angRes, w, angRes)
        shuffle_out = x_view.permute(0, 1, 4, 2, 5, 6, 3, 7).contiguous()
        return shuffle_out.view(b, c_o, h_o * angRes, w_o * angRes)
    else:
        bsz = int(1 / scale_factor)
        x_view = x.contiguous().view(b, c, h_o, bsz, angRes, w_o, bsz, angRes)
        # 0, 1, 2,   3,   4,  5,   6,   7
        shuffle_out = x_view.permute(0, 1, 3, 6, 2, 4, 5, 7).contiguous()
        # b, c, bsz, bsz, angRes, h_0, angRes, w_o
        return shuffle_out.view(b, c_o, h_o * angRes, w_o * angRes)


class BlockShuffle(nn.Module):
    def __init__(self, scale_factor, angRes):
        super(BlockShuffle, self).__init__()
        self.scale_factor = scale_factor
        self.angRes = angRes

    def forward(self, x):
        return block_shuffle(x, self.scale_factor, self.angRes)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


# class ReflectionPad2d_lf(nn.Module):
#     def __init__(self, angRes, padding):
#         super(ReflectionPad2d_lf, self).__init__()
#         self.angRes = angRes
#         self.pad = nn.ReflectionPad2d(padding)
#         self.dw_shuffle = PixelShuffle(1 / angRes)
#         self.up_shuffle = PixelShuffle(angRes)

#     def forward(self, x):
#         x = self.dw_shuffle(x)
#         x = self.pad(x)
#         x = self.up_shuffle(x)
#         return x


class ConvNorm_dilated(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 kernel_size,
                 angRes,
                 stride=1,
                 norm=False):
        super(ConvNorm_dilated, self).__init__()

        # reflection_padding = kernel_size // 2
        # reflection_padding = (kernel_size - 1) // 2
        # self.reflection_pad = ReflectionPad2d_lf(angRes, reflection_padding)
        self.conv = nn.Conv2d(in_feat,
                              out_feat,
                              stride=stride,
                              kernel_size=kernel_size,
                              dilation=angRes,
                              bias=True,
                              padding=angRes)

        # self.norm = norm
        # if norm == 'IN':
        #     self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        # elif norm == 'BN':
        #     self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        # out = self.reflection_pad(x)
        out = self.conv(x)
        # if self.norm:
        #     out = self.norm(out)
        return out


class ResidualGroup_dilated(nn.Module):
    def __init__(self,
                 Block,
                 n_resblocks,
                 n_feat,
                 kernel_size,
                 reduction,
                 act,
                 norm=False,
                 angRes=5):
        super(ResidualGroup_dilated, self).__init__()

        modules_body = [
            Block(n_feat,
                  n_feat,
                  kernel_size,
                  reduction,
                  bias=True,
                  norm=norm,
                  act=act,
                  angRes=angRes) for _ in range(n_resblocks)
        ]
        modules_body.append(
            ConvNorm_dilated(n_feat,
                             n_feat,
                             kernel_size,
                             angRes,
                             stride=1,
                             norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Dilated_Global_AvgPool2d(nn.Module):
    def __init__(self, dilation):
        super(Dilated_Global_AvgPool2d, self).__init__()
        self.dw_shuffle = PixelShuffle(1 / dilation)
        self.up_shuffle = PixelShuffle(dilation)
        self.global_pool2d = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_down = self.dw_shuffle(x)
        x_down = self.global_pool2d(x_down)
        return self.up_shuffle(x_down)


class CALayer_dilated(nn.Module):
    def __init__(self, angRes, channel, reduction=16):
        super(CALayer_dilated, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dilated_avg_pool = Dilated_Global_AvgPool2d(dilation=angRes)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())
        self.up_shuffle = PixelShuffle(angRes)
        self.dw_shuffle = PixelShuffle(1 / angRes)

    def forward(self, x):
        # y = self.avg_pool(x)
        # y = self.conv_du(y)
        # return x * y
        y = self.dilated_avg_pool(x)
        y = self.conv_du(y)
        return self.up_shuffle(self.dw_shuffle(y) * self.dw_shuffle(x))


class RCAB_dilated(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 kernel_size,
                 reduction,
                 angRes,
                 bias=True,
                 norm=False,
                 act=nn.ReLU(True),
                 downscale=False):
        super(RCAB_dilated, self).__init__()
        self.body = nn.Sequential(
            ConvNorm_dilated(in_feat,
                             out_feat,
                             kernel_size,
                             angRes=angRes,
                             stride=1,
                             norm=norm),
            act,
            ConvNorm_dilated(out_feat,
                             out_feat,
                             kernel_size,
                             angRes=angRes,
                             stride=1,
                             norm=norm),
            CALayer_dilated(angRes, out_feat,
                            reduction)  # 正常模型使用CALayer_dilated
            # CALayer(out_feat, reduction)  # for ACA's ablation study
        )
        self.downscale = downscale

    def forward(self, x):
        res = x
        out = self.body(x)
        out += res
        return out


class Interpolation_dilated(nn.Module):
    def __init__(self,
                 n_resgroups,
                 n_resblocks,
                 n_feats,
                 angRes,
                 reduction=16,
                 act=nn.LeakyReLU(0.2, True),
                 norm=False):
        super(Interpolation_dilated, self).__init__()

        # define modules: head, body, tail
        self.headConv = nn.Conv2d(n_feats * 2,
                                  n_feats,
                                  kernel_size=3,
                                  stride=1,
                                  dilation=angRes,
                                  padding=angRes,
                                  bias=True)
        modules_body = [
            ResidualGroup_dilated(
                # RCAB,
                RCAB_dilated,
                n_resblocks=n_resblocks,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction,
                act=act,
                norm=norm,
                angRes=angRes) for _ in range(n_resgroups)
        ]
        self.body = nn.Sequential(*modules_body)

        # self.tailConv = conv3x3(n_feats, n_feats)
        self.tailConv = nn.Conv2d(n_feats,
                                  n_feats,
                                  kernel_size=3,
                                  stride=1,
                                  dilation=angRes,
                                  padding=angRes,
                                  bias=True)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat((x0, x1), dim=1)  # [b,192*2,ah/8,aw/8]
        x = self.headConv(x)  # [b,192,ah/8,aw/8]

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        return out


class Encoder(nn.Module):
    def __init__(self, angRes, in_channels=3, depth=3):
        super(Encoder, self).__init__()
        self.angRes = angRes
        # Shuffle pixels to expand in channel dimension
        self.shuffler = BlockShuffle(
            1 / 2**depth, angRes)  # 1 / 2 ** depth represent chiduyinzi

        lrelu = nn.LeakyReLU(0.2, True)

        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation_dilated(5,
                                                 12,
                                                 in_channels * (4**depth),
                                                 angRes=angRes,
                                                 act=lrelu)

    def forward(self, x1, x2):
        """
        x1, x2.shape = [b,c,ah,aw]
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)  # [b,c*alpha**2,h/alpha,w/alpha]
        feats2 = self.shuffler(x2)  # [b,c*alpha**2,h/alpha,w/alpha]
        feats = self.interpolate(feats1, feats2)
        return feats


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = CharbonnierLoss()

    def forward(self, SR, HR):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):

    pass


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


if __name__ == '__main__':
    from thop import profile
    angRes_in = 3
    angRes_out = 5
    h = 256
    w = 256
    input_pre = torch.randn([1, 3, angRes_in * h, angRes_in * w]).cuda()
    input_nxt = torch.randn([1, 3, angRes_in * h, angRes_in * w]).cuda()
    label = torch.randn([1, 3, angRes_out * h, angRes_out * w]).cuda()
    model = get_model({"angRes_in": angRes_in, "angRes_out": angRes_out})
    model = model.cuda()

    # 计算#params和#flops
    # total_ops, total_params = profile(model, (input_pre, input_nxt))
    # print('Number of parameters: %.2fM' % (total_params / 1e6))
    # print('Number of FLOPs: %.2fG' % (total_ops * 2 / 1e9))

    # 测试显存占用
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad is True], lr=1e-4)
    loss_func = get_loss({})
    for i in range(50):
        with autocast():
            output = model(input_pre, input_nxt)
            torch.cuda.synchronize()
            loss = loss_func(output, label)
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # 6478MB
    #     print(f"loss: {loss.item():.5f}")
    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    #     with autocast():
    #         output = model(input_pre, input_nxt)
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    # 406G
    # 43.36M
