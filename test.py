import importlib
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils.utils import (LFdivide, LFintegrate, Logger, cal_metrics,
                         create_dir, make_coord, rgb2ycbcr)
from utils.utils_datasets import MultiTestSetDataLoader

import cv2

import os


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

    # num_params = 0
    # for name, param in net.named_parameters():
    #     if hasattr(param, 'shape'):
    #         num_params += np.prod(list(param.shape))
    # print("# of params is [{}]".format(num_params))
    # net.cuda()
    # from ptflops import get_model_complexity_info
    # def prepare_input(resolution):
    #     inp = torch.randn(1, 3, 5*32, 5*32).cuda()
    #     cell = torch.randn(1, 64, 64, 2).cuda()
    #     coord = torch.randn(1, 64, 64, 2).cuda()

    #     return dict(inp=inp, coord=coord, cell=cell)
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(
    #         model=net,
    #         input_res=(3, 224, 224),
    #         as_strings=True,
    #         input_constructor=prepare_input,
    #     )
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # exit(0)

    # exit(0)
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
    net.eval()

    psnr_iter_test = []
    ssim_iter_test = []

    for idx_iter, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):

        data = data_batch['inp']  # low resolution
        label = data_batch['gt']  # high resolution
        assert label.shape[0] == 1 and label.ndim == 4
        data = data.squeeze(0)  # [3,uh,vw]
        label = label.squeeze(0)  # [3,uh,vw]
        label = rgb2ycbcr(label.permute(1, 2, 0).contiguous().numpy())[..., 0]  # y channel with [uh,vw] shape.
        label = torch.from_numpy(label).cuda(args.local_rank)  # [uh,vw]
        n_colors, uh, vw = data.shape

        h0, w0 = int(uh//args.angRes), int(vw//args.angRes)

        subLFin = LFdivide(data, args.angRes, args.patch_size_for_test, args.stride_for_test)
        subLFin = subLFin.cuda(args.local_rank)
        numU, numV, n_colors, H, W = subLFin.size()
        # args.scale_factor = 1.5
        subLFout = torch.zeros(numU, numV, n_colors, int(args.angRes * args.patch_size_for_test * args.scale_factor),
                               int(args.angRes * args.patch_size_for_test * args.scale_factor))
        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u:u+1, v:v+1, :, :, :]  # [1,1,3,uh,vw]
                tmp = tmp.squeeze(0)

                hr_coord = make_coord(
                    [int((tmp.shape[-2] // args.angRes) * args.scale_factor),
                    int((tmp.shape[-1] // args.angRes) * args.scale_factor)],
                    flatten=False
                )  # [h',w',2]
                hr_coord = hr_coord.unsqueeze(0)
                hr_coord = hr_coord.to(device)

                cell = torch.ones_like(hr_coord)
                cell[:, 0] *= 2 / ((tmp.shape[-2] // args.angRes) * args.scale_factor)  # 一个cell的高
                cell[:, 1] *= 2 / ((tmp.shape[-1] // args.angRes) * args.scale_factor)  # 一个cell的宽
                cell = cell.to(device)

                out = net(tmp, hr_coord, cell)

                subLFout[u:u+1, v:v+1, :, :, :] = out.squeeze(0)

        Sr_4D_rgb = LFintegrate(
            subLFout,
            args.angRes,
            int(args.patch_size_for_test * args.scale_factor),
            int(args.stride_for_test * args.scale_factor),
            int(h0 * args.scale_factor),
            int(w0 * args.scale_factor)
        )  # [u,v,3,h,w]
        Sr_SAI_rgb = Sr_4D_rgb.permute(0, 3, 1, 4, 2).reshape((int(h0 * args.angRes * args.scale_factor),
                                                          int(w0 * args.angRes * args.scale_factor),
                                                          n_colors))  # [uh,vw,3]
        Sr_SAI_rgb_uint8 = (Sr_SAI_rgb.detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype('uint8')

        # save predicted SAI.
        os.makedirs(f"./Figs/SR_{args.scale_factor:.1f}x", exist_ok=True)
        output_path = f"./Figs/SR_{args.scale_factor:.1f}x/{test_loader.dataset.file_list[idx_iter].replace('/', '_')}.png"
        cv2.imwrite(output_path, Sr_SAI_rgb_uint8[..., ::-1])
        if args.scale_factor != 2:
            continue
        Sr_SAI_y = rgb2ycbcr(Sr_SAI_rgb.detach().numpy())[..., 0]  # [uh,vw]
        Sr_SAI_y = torch.from_numpy(Sr_SAI_y).cuda(args.local_rank)
        psnr, ssim = cal_metrics(args, label, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass
    # exit(0)
    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    torch.cuda.empty_cache()
    return psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    from option import args

    main(args)
