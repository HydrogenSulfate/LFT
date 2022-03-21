import importlib
from collections import OrderedDict
# from contextlib import nullcontext

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import (LFdivide, LFintegrate, Logger,
                         WarmUpCosineAnnealingLR, cal_metrics, create_dir,
                         make_coord, rgb2ycbcr)
from utils.utils_datasets import TestSetDataLoader, TrainSetDataLoader, InferSetDataLoader


def main(args):
    """[summary]

    Args:
        args ([type]): [description]
    """
    ''' Create Dir for Save '''
    experiment_dir, checkpoints_dir, log_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda '''
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # device = torch.device("cpu", args.local_rank)

    ''' DATA TRAINING LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    infer_Dataset = InferSetDataLoader(args, args.data_name)
    train_sampler = None
    if_shuffle = True
    infer_loader = torch.utils.data.DataLoader(
        dataset=infer_Dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=if_shuffle,
        drop_last=True,
        sampler=train_sampler
    )

    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' load pre-trained pth '''
    if args.use_pre_pth is True:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('resume training and load [1.model] state dict')
            except Exception as e:
                logger.log_string(str(e))
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
        except Exception as e:
            logger.log_string(str(e))
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass

    local_rank = args.local_rank
    net = net.to(local_rank)

    ''' VISUALIZING '''
    if args.use_pre_pth:
        del checkpoint
    logger.log_string('\nStart Visualizing...')

    vis(local_rank, infer_loader, device, net)


@torch.no_grad()
def vis(local_rank, infer_loader, device, net):
    """[summary]

    Args:
        local_rank ([type]): [description]
        infer_loader ([type]): [description]
        device ([type]): [description]
        net ([type]): [description]
        criterion ([type]): [description]
    """
    net.eval()

    for idx_iter, data_batch in tqdm(enumerate(infer_loader), total=len(infer_loader), ncols=70):

        data = data_batch['inp'].to(device)      # low resolution
        coord = data_batch['coord'].to(device)
        cell = data_batch['cell'].to(device)

        data = (data - 0.5) / 0.5  # 归一化到坐标值域的[-1,1]区间
        # print(data.shape)
        # print(coord.shape)
        # print(cell.shape)
        # exit(0)
        out = net(data, coord, cell)


if __name__ == '__main__':
    from option import args

    main(args)
