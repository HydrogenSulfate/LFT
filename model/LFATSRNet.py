import torch
import torch.nn as nn
from utils.utils import MacPI2SAI, SAI2MacPI

# import torch.nn.functional as F
# from einops import rearrange


class get_model(nn.Module):
    """LFATSRNet init

    Args:
        angRes_in (int, optional): angRes input. Defaults to 3.
        angRes_out (int, optional): angRes output. Defaults to 5.
    """

    def __init__(self, args: dict):
        super(get_model, self).__init__()
        self.encoder = Encoder(args.angRes_in, in_channels=3, depth=3)
        self.decoder = Decoder(args.angRes_in, depth=3)
        self.AngSRNet = AngularSR_Net(inter_plane=64, ain=3, aout=5, n_resblocks=7)

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
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

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
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y  # , y


class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, norm=False, act=nn.ReLU(True), downscale=False,
                 return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
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
    def __init__(self, inter_plane, ain, aout, n_resblocks):
        super(AngularSR_Net, self).__init__()
        self.ain = ain
        self.aout = aout
        self.PixelDown = PixelShuffle(1 / ain)
        self.PixelUp = PixelShuffle(aout)
        self.BicubicUp = nn.Upsample(size=(aout, aout), mode='bicubic', align_corners=True)
        self.pre_conv = nn.Conv2d(3 * ain * ain, inter_plane, kernel_size=3, padding=1)
        self.tail_conv = nn.Conv2d(inter_plane, 3 * aout * aout, kernel_size=3, padding=1)

        modules_body = [RCAB(inter_plane, inter_plane, 3, reduction=16, norm=False, act=nn.LeakyReLU(0.2, inplace=True))
                        for _ in range(n_resblocks)]
        # modules_body = [RCAMB(inter_plane, inter_plane, 3, norm=False, act=nn.LeakyReLU(0.2, inplace=True))
        #                 for _ in range(n_resblocks)]
        self.n_resb = nn.Sequential(*modules_body)

    def forward(self, x_lr):
        b, c, ah, aw = x_lr.size()
        h, w = ah // self.ain, aw // self.ain
        # bicubic upsample for Angular
        x_hr = x_lr.view(b, c, h, self.ain, w, self.ain)
        x_hr = x_hr.permute(0, 1, 2, 4, 3, 5)  # [b,c,h,w,a,a]
        x_hr = x_hr.reshape(b, c * h * w, self.ain, self.ain)  # [b,cxhxw,a,a]
        x_hr = self.BicubicUp(x_hr)  # [b,cxhxw,a',a']
        x_hr = x_hr.view(b, c, h, w, self.aout, self.aout)  # [b,c,h,w,a',a']
        x_hr = x_hr.permute(0, 1, 2, 4, 3, 5)  # [b,c,h,a',w,a']
        x_hr = x_hr.reshape(b, c, h * self.aout, w * self.aout)  # [b,c,hxa',wxa']

        # detail restore
        x_res = self.PixelDown(x_lr)
        x_res = self.pre_conv(x_res)
        x_res = self.n_resb(x_res)
        x_res = self.tail_conv(x_res)
        x_res = self.PixelUp(x_res)
        x_hr += x_res
        return x_hr


class Decoder(nn.Module):
    def __init__(self, angRes, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = BlockShuffle(2 ** depth, angRes)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()  # [b,c,h,w]

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        #  [b,c,h,w]->[b,c,h',bh,w',bw]
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()  # [b,c,bh,bw,h',w']

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)  # [b,c*bh*bw,h',w']


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
        x_view = x.contiguous().view(b, c_o, scale_factor, scale_factor, h, angRes, w, angRes)
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


class ReflectionPad2d_lf(nn.Module):
    def __init__(self, angRes, padding):
        super(ReflectionPad2d_lf, self).__init__()
        self.angRes = angRes
        self.pad = nn.ReflectionPad2d(padding)
        self.dw_shuffle = PixelShuffle(1 / angRes)
        self.up_shuffle = PixelShuffle(angRes)

    def forward(self, x):
        x = self.dw_shuffle(x)
        x = self.pad(x)
        x = self.up_shuffle(x)
        return x


class ConvNorm_dilated(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, angRes, stride=1, norm=False):
        super(ConvNorm_dilated, self).__init__()

        # reflection_padding = kernel_size // 2
        reflection_padding = (kernel_size - 1) // 2
        self.reflection_pad = ReflectionPad2d_lf(angRes, reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, dilation=angRes, bias=True)

        # self.norm = norm
        # if norm == 'IN':
        #     self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        # elif norm == 'BN':
        #     self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        # if self.norm:
        #     out = self.norm(out)
        return out


class ResidualGroup_dilated(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False, angRes=5):
        super(ResidualGroup_dilated, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act, angRes=angRes)
                        for _ in range(n_resblocks)]
        modules_body.append(ConvNorm_dilated(n_feat, n_feat, kernel_size, angRes, stride=1, norm=norm))
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
            nn.Sigmoid()
        )
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
    def __init__(self, in_feat, out_feat, kernel_size, reduction, angRes, bias=False, norm=False,
                 act=nn.LeakyReLU(0.2, True),
                 downscale=False):
        super(RCAB_dilated, self).__init__()
        self.body = nn.Sequential(
            ConvNorm_dilated(in_feat, out_feat, kernel_size, angRes=angRes, stride=1, norm=norm),
            act,
            ConvNorm_dilated(out_feat, out_feat, kernel_size, angRes=angRes, stride=1, norm=norm),
            CALayer_dilated(angRes, out_feat, reduction)  # 正常模型使用CALayer_dilated
            # CALayer(out_feat, reduction)  # for ACA's ablation study
        )
        self.downscale = downscale

    def forward(self, x):
        res = x
        out = self.body(x)
        out += res
        return out


class Interpolation_dilated(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats,
                 angRes, reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation_dilated, self).__init__()

        # define modules: head, body, tail
        self.headConv = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1,
                                  dilation=angRes, padding=angRes, bias=True)
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
                angRes=angRes) for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        # self.tailConv = conv3x3(n_feats, n_feats)
        self.tailConv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, dilation=angRes, padding=angRes * (3 - 1) // 2,
                                  bias=True)

    def forward(self, x0, x1, return_x=False):
        # Build input tensor
        x = torch.cat((x0, x1), dim=1)  # [b,192*2,ah/8,aw/8]
        x = self.headConv(x)  # [b,192,ah/8,aw/8]

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        if return_x:
            return out, x
        return out


class Encoder(nn.Module):

    def __init__(self, angRes, in_channels=3, depth=3):
        super(Encoder, self).__init__()
        self.angRes = angRes
        # Shuffle pixels to expand in channel dimension
        self.shuffler = BlockShuffle(1 / 2 ** depth, angRes)  # 1 / 2 ** depth represent chiduyinzi

        lrelu = nn.LeakyReLU(0.2, True)

        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation_dilated(5, 12, in_channels * (4 ** depth), angRes=angRes, act=lrelu)

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


if __name__ == '__main__':
    from thop import profile
    angRes_in = 3
    angRes_out = 5
    h = 128
    w = 128
    input_pre = torch.randn([1, 3, angRes_in * h, angRes_in * w]).cuda()
    input_nxt = torch.randn([1, 3, angRes_in * h, angRes_in * w]).cuda()
    # label = torch.randn([1, 3, angRes_out * h, angRes_out * w])
    model = get_model({
        "angRes_in": angRes_in,
        "angRes_out": angRes_out
    })
    model = model.cuda()
    total_ops, total_params = profile(model, (input_pre, input_nxt))
    print('Number of parameters: %.2fM' % (total_params / 1e6))
    print('Number of FLOPs: %.2fG' % (total_ops * 2 / 1e9))
