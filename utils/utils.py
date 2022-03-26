import logging
import math
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from einops import rearrange
from skimage import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_logger(log_dir, args):
    ''' LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_dir(args):
    experiment_dir = Path(args.path_log)
    experiment_dir.mkdir(exist_ok=True)
    task_path = 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + str(args.scale_factor) + 'x'

    experiment_dir = experiment_dir.joinpath(task_path)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.model_name)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.data_name)
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    return experiment_dir, checkpoints_dir, log_dir


class Logger(object):
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, string):
        if args.local_rank <= 0:
            self.logger.info(string)
            print(string)


def cal_metrics(angRes: int, label: torch.Tensor, out: torch.Tensor) -> Tuple[float, float]:
    if len(label.size()) == 2:
        label = rearrange(label, '(a1 h) (a2 w) -> 1 1 a1 h a2 w', a1=angRes, a2=angRes)
        out = rearrange(out, '(a1 h) (a2 w) -> 1 1 a1 h a2 w', a1=angRes, a2=angRes)

    if len(label.size()) == 4:
        [B, C, H, W] = label.size()
        label = label.view((B, C, angRes, H // angRes, angRes, H // angRes))
        out = out.view((B, C, angRes, H // angRes, angRes, W // angRes))

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_rgb = label.detach().cpu()
    out_rgb = out.detach().cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_rgb[b, :, u, :, v, :].numpy().transpose(1, 2, 0),
                                                                out_rgb[b, :, u, :, v, :].numpy().transpose(1, 2, 0))

                SSIM[b, u, v] = metrics.structural_similarity(label_rgb[b, :, u, :, v, :].numpy().transpose(1, 2, 0),
                                                              out_rgb[b, :, u, :, v, :].numpy().transpose(1, 2, 0),
                                                              gaussian_weights=True,
                                                              multichannel=True)
    if np.sum(PSNR > 0) == 0:
        PSNR_mean = 0.0
    else:
        PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)

    if np.sum(SSIM > 0) == 0:
        SSIM_mean = 0.0
    else:
        SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)
    return PSNR_mean, SSIM_mean


def LFdivide(data, angRes, patch_size, stride):
    n_colors, uh, vw = data.shape
    h0 = uh // angRes  # 512
    w0 = vw // angRes  # 512
    bdr = (patch_size - stride) // 2  # 8
    h = h0 + 2 * bdr  # 512+16=528，填充后的子光圈图像高度
    w = w0 + 2 * bdr  # 512+16=528，填充后的子光圈图像宽度
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1 # 32
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1 # 32
    hE = stride * (numU - 1) + patch_size  # 528
    wE = stride * (numV - 1) + patch_size  # 528

    dataE = torch.zeros(n_colors, hE * angRes, wE * angRes)  # [3,528*5,528*5]
    for u in range(angRes):
        for v in range(angRes):
            Im = data[:, u * h0:(u + 1) * h0, v * w0:(v + 1) * w0]  # 遍历每一个子光圈图像[c,h,w]
            dataE[:, u * hE:u * hE + h, v * wE:v * wE + w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, n_colors, patch_size * angRes, patch_size * angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u * hE + kh * stride
                    vv = v * wE + kw * stride
                    subLF[kh, kw, :, u * patch_size:(u + 1) * patch_size, v * patch_size:(v + 1) * patch_size] = \
                        dataE[:, uu:uu + patch_size, vv:vv + patch_size]
    return subLF


def ImageExtend(Im: torch.Tensor, bdr: int) -> torch.Tensor:
    """ReflectPad2d(im, bdr)，但是不共用边界

    Args:
        Im (torch.Tensor): 输入图像
        bdr (int): 边界填充宽度

    Returns:
        torch.Tensor: xxx.
    """
    n_colors, h, w = Im.shape  # 3,544,544
    Im_lr = torch.flip(Im, dims=[-1])  # [3,h,w]
    Im_ud = torch.flip(Im, dims=[-2])  # [3,h,w]
    Im_diag = torch.flip(Im, dims=[-1, -2])  # [3,h,w]

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)  # [3,h,3w]
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)  # [3,h,3w]
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)  # [3,h,3w]
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)  # [3,3h,3w]
    Im_out = Im_Ext[:, h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]  # []

    return Im_out


def LFintegrate(subLF: torch.Tensor, angRes: int, pz: int, stride: int, h0: int, w0: int) -> torch.Tensor:
    numU, numV, n_colors, pH, pW = subLF.shape
    # H, W = numU*pH, numV*pW
    ph, pw = pH // angRes, pW // angRes
    bdr = (pz - stride) // 2
    temp = torch.zeros(n_colors, stride * numU, stride * numV)
    outLF = torch.zeros(angRes, angRes, n_colors, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[:, ku * stride:(ku + 1) * stride, kv * stride:(kv + 1) * stride] = \
                        subLF[ku, kv, :, u * ph + bdr:u * ph + bdr + stride, v * pw + bdr:v * ph + bdr + stride]

            outLF[u, v, :, :, :] = temp[:, 0:h0, 0:w0]

    return outLF


def rgb2ycbcr(x):
    """
    x: [h, w, c(r,g,b)]
    """
    y = np.zeros(x.shape, dtype='double')
    # y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 / 255
    y[:, :, 0] = 65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] + 24.966 * x[:, :, 2] + 16.0
    y[:, :, 1] = -37.797 * x[:, :, 0] - 74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:, :, 2] = 112.000 * x[:, :, 0] - 93.786 * x[:, :, 1] - 18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()  # -h+1~h-1
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def ycbcr2rgb(x):
    """
    x: [h, w, c(y,cb,cr)]
    """
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat) * 255

    y = np.zeros(x.shape, dtype='double')
    y[:, :, 0] = mat_inv[0, 0] * x[:, :, 0] + mat_inv[0, 1] * x[:, :, 1] + mat_inv[0, 2] * x[:, :, 2] - 16.0 / 255.0
    y[:, :, 1] = mat_inv[1, 0] * x[:, :, 0] + mat_inv[1, 1] * x[:, :, 1] + mat_inv[1, 2] * x[:, :, 2] - 128.0 / 255.0
    y[:, :, 2] = mat_inv[2, 0] * x[:, :, 0] + mat_inv[2, 1] * x[:, :, 1] + mat_inv[2, 2] * x[:, :, 2] - 128.0 / 255.0

    return y


def crop_center_view(data, angRes_in, angRes_out):
    assert angRes_in >= angRes_out, 'angRes_in requires to be greater than angRes_out'
    [B, _, H, W] = data.size()
    patch_size = H // angRes_in
    data = data[:, :,
                (angRes_in - angRes_out) // 2 * patch_size:(angRes_in + angRes_out) // 2 * patch_size,
                (angRes_in - angRes_out) // 2 * patch_size:(angRes_in + angRes_out) // 2 * patch_size]

    return data


def cal_loss_class(probability):
    assert len(probability.size()) == 2, 'probability requires a 2-dim tensor'
    [B, num_cluster] = probability.size()
    loss_class = 0
    for batch in range(B):
        sum_re = 0
        for i in range(num_cluster - 1):
            for j in range(i + 1, num_cluster):
                sum_re += abs(probability[batch][i] - probability[batch][j])

        loss_class += ((num_cluster - 1) - sum_re)
    loss_class = loss_class / B

    return loss_class


class WarmUpCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, warm_multiplier, warm_duration, cos_duration, eta_min=0, last_epoch=-1):
        assert warm_duration >= 0
        assert warm_multiplier > 1.0
        self.warm_m = float(warm_multiplier)
        self.warm_d = warm_duration
        self.cos_duration = cos_duration
        self.cos_eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, self.cos_duration, eta_min, last_epoch)

    def get_lr(self):
        if self.warm_d == 0:
            return super(WarmUpCosineAnnealingLR, self).get_lr()
        else:
            if not self._get_lr_called_within_step:
                warnings.warn("To get the last learning rate computed by the scheduler, "
                              "please use `get_last_lr()`.", UserWarning)
            if self.last_epoch == 0:
                return [lr / self.warm_m for lr in self.base_lrs]
                # return self.base_lrs / self.warm_m
            elif self.last_epoch <= self.warm_d:
                return [(self.warm_d + (self.warm_m - 1) * self.last_epoch) / (self.warm_d + (self.warm_m - 1) * (self.last_epoch - 1)) * group['lr'] for group in self.optimizer.param_groups]
            else:
                cos_last_epoch = self.last_epoch - self.warm_d
                if cos_last_epoch == 0:
                    return self.base_lrs
                elif (cos_last_epoch - 1 - self.cos_duration) % (2 * self.cos_duration) == 0:
                    return [group['lr'] + (base_lr - self.cos_eta_min) *
                            (1 - math.cos(math.pi / self.cos_duration)) / 2
                            for base_lr, group in
                            zip(self.base_lrs, self.optimizer.param_groups)]
                return [(1 + math.cos(math.pi * cos_last_epoch / self.cos_duration)) /
                        (1 + math.cos(math.pi * (cos_last_epoch - 1) / self.cos_duration)) *
                        (group['lr'] - self.cos_eta_min) + self.cos_eta_min
                        for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.warm_d == 0:
            return super(WarmUpCosineAnnealingLR, self)._get_closed_form_lr()
        else:
            if self.last_epoch <= self.warm_d:
                return [base_lr * (self.warm_d + (self.warm_m - 1) * self.last_epoch) / (self.warm_d * self.warm_m) for base_lr in self.base_lrs]
            else:
                cos_last_epoch = self.last_epoch - self.warm_d
                return [self.cos_eta_min + (base_lr - self.cos_eta_min) *
                    (1 + math.cos(math.pi * cos_last_epoch / self.cos_duration)) / 2
                    for base_lr in self.base_lrs]


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
    a=5
    h = w = 512
    data = np.zeros([3, h*a, w*a])
    x = LFdivide(data, a, 32, 16)
