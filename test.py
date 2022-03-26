import importlib
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import LFdivide, LFintegrate, Logger, cal_metrics, create_dir
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
def test(test_loader: DataLoader, device: torch.device, net: torch.Module) -> Tuple[float, float]:
    net.eval()

    psnr_iter_test = []
    ssim_iter_test = []

    for idx_iter, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):

        pre: torch.Tensor = data_batch['pre'].to(device)      # low resolution
        next: torch.Tensor = data_batch['nxt'].to(device)      # low resolution
        label: torch.Tensor = data_batch['gt'].to(device)      # high resolution

        assert label.shape[0] == 1 and label.ndim == 4

        N, n_colors, uh, vw = label.shape

        h0, w0 = int(uh//args.angRes_in), int(vw//args.angRes_in)

        subLFin_pre = LFdivide(pre, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        subLFin_nxt = LFdivide(next, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        subLFin_pre = subLFin_pre.cuda(args.local_rank)
        subLFin_nxt = subLFin_nxt.cuda(args.local_rank)
        numU, numV, n_colors, H, W = subLFin_pre.size()

        subLFout = torch.zeros(
            numU,
            numV,
            n_colors,
            args.angRes_out * args.patch_size_for_test,
            args.angRes_out * args.patch_size_for_test
        )
        for u in range(numU):
            for v in range(numV):
                tmp_pre = subLFin_pre[u:u+1, v:v+1, :, :, :]  # [1,1,3,uh,vw]
                tmp_pre = tmp_pre.squeeze(0)

                tmp_nxt = subLFin_nxt[u:u+1, v:v+1, :, :, :]  # [1,1,3,uh,vw]
                tmp_nxt = tmp_nxt.squeeze(0)

                out = net(tmp_pre, tmp_nxt)

                subLFout[u:u+1, v:v+1, :, :, :] = out

        Sr_4D_rgb = LFintegrate(
            subLFout,
            args.angRes_out,
            args.patch_size_for_test,
            args.stride_for_test,
            h0,
            w0
        )  # [u,v,3,h,w]
        Sr_SAI_rgb = Sr_4D_rgb.permute(0, 3, 1, 4, 2).reshape((h0 * args.angRes * args.scale_factor,
                                                          w0 * args.angRes * args.scale_factor,
                                                          n_colors))  # [uh,vw,3]
        psnr, ssim = cal_metrics(args, label, Sr_SAI_rgb)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    torch.cuda.empty_cache()
    return psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    from option import args

    main(args)
