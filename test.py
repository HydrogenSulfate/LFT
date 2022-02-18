import importlib
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils.utils import (LFdivide, LFintegrate, Logger, cal_metrics,
                         create_dir, make_coord, rgb2ycbcr)
from utils.utils_datasets import MultiTestSetDataLoader


def main(args):
    ''' Create Dir for Save'''
    experiment_dir, checkpoints_dir, log_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda '''
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # device = torch.device("cpu", args.local_rank)

    ''' DATA TEST LOADING '''
    logger.log_string('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of test data is: %d" % length_of_tests)

    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' load pre-trained pth '''
    ckpt_path = args.path_pre_pth
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # start_epoch = checkpoint['epoch']
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        logger.log_string('Use pretrain model!')
    except Exception as e:
        logger.log_string(str(e))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)
        logger.log_string('Use pretrain model!')

    net = net.to(device)
    cudnn.benchmark = True

    ''' TEST on every dataset'''
    logger.log_string('\nStart test...')
    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            psnr_epoch_test, ssim_epoch_test = test(test_loader, device, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            logger.log_string('Test on %s, psnr/ssim is %.2f/%.3f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        pass
    pass


@torch.no_grad()
def test(test_loader, device, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        # Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        # Hr_SAI_y = Hr_SAI_y.squeeze()

        data = data_batch['inp'].to(device)  # low resolution
        label = data_batch['gt']  # high resolution
        assert label.shape[0] == 1 and label.ndim == 4
        # coord = data_batch['coord'].to(device)
        # cell = data_batch['cell'].to(device)

        data = data.squeeze(0)  # [3,uh,vw]
        label = label.squeeze(0)  # [3,uh,vw]
        label = rgb2ycbcr(label.permute(1, 2, 0).contiguous().numpy())[..., 0]  # y channel with [uh,vw] shape.
        label = torch.from_numpy(label).cuda(args.local_rank)  # [uh,vw]
        n_colors, uh, vw = data.shape

        h0, w0 = int(uh // args.angRes), int(vw // args.angRes)

        subLFin = LFdivide(data, args.angRes, args.patch_size_for_test, args.stride_for_test)
        numU, numV, n_colors, H, W = subLFin.size()
        subLFout = torch.zeros(numU, numV, n_colors, args.angRes * args.patch_size_for_test * args.scale_factor,
                               args.angRes * args.patch_size_for_test * args.scale_factor)
        subLFin = subLFin.cuda(args.local_rank)
        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u:u + 1, v:v + 1, :, :, :]  # [1,1,3,uh,vw]
                tmp = tmp.squeeze(0)
                # with torch.no_grad():
                net.eval()
                # torch.cuda.empty_cache()
                hr_coord = make_coord([(tmp.shape[-2] // args.angRes) * args.scale_factor, (tmp.shape[-1] // args.angRes) * args.scale_factor], flatten=False)  # [h',w',2]
                hr_coord = hr_coord.unsqueeze(0)
                hr_coord = hr_coord.to(device)
                cell = torch.ones_like(hr_coord)
                cell[:, 0] *= 2 / ((tmp.shape[-2] // args.angRes)*args.scale_factor)  # 一个cell的高
                cell[:, 1] *= 2 / ((tmp.shape[-1] // args.angRes)*args.scale_factor)  # 一个cell的宽
                cell = cell.to(device)
                out = net(tmp, hr_coord, cell)
                subLFout[u:u + 1, v:v + 1, :, :, :] = out.squeeze(0)

        Sr_4D_y = LFintegrate(subLFout, args.angRes, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, h0 * args.scale_factor,
                              w0 * args.scale_factor)  # [u,v,3,h,w]
        Sr_SAI_y = Sr_4D_y.permute(0, 3, 1, 4, 2).reshape((h0 * args.angRes * args.scale_factor,
                                                          w0 * args.angRes * args.scale_factor,
                                                          n_colors))  # [uh,vw,3]
        Sr_SAI_y = rgb2ycbcr(Sr_SAI_y.detach().numpy())[..., 0]  # [uh,vw]
        Sr_SAI_y = torch.from_numpy(Sr_SAI_y).cuda(args.local_rank)
        psnr, ssim = cal_metrics(args, label, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    from option import args

    main(args)
