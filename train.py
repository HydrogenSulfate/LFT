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
from utils.utils_datasets import TestSetDataLoader, TrainSetDataLoader


def main(args):
    ''' Create Dir for Save '''
    experiment_dir, checkpoints_dir, log_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' NCCL '''
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    parallel = world_size != 1

    ''' CPU or Cuda '''
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # device = torch.device("cpu", args.local_rank)

    ''' DATA TRAINING LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    if parallel:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_Dataset)
        if_shuffle = False
    else:
        train_sampler = None
        if_shuffle = True
    iters_to_accumulate = args.global_batch_size // (args.batch_size * world_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=if_shuffle, drop_last=True, sampler=train_sampler)

    # train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
    #                                            batch_size=args.batch_size, shuffle=True,)

    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' load pre-trained pth '''
    if args.use_pre_pth is False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pretrain model!')
    elif dist.get_rank() == 0:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
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
        except Exception as e:
            logger.log_string(str(e))
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass

    local_rank = args.local_rank
    net = net.to(local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)

    '''LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)

    ''' optimizer'''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad is True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    # no_wd_list = ['norm.weight', 'norm.bias']
    # no_wd_params = [p for n, p in net.named_parameters() if p.requires_grad and any([key in n for key in no_wd_list])]
    # no_wd_params_name = [n for n, p in net.named_parameters() if p.requires_grad and any([key in n for key in no_wd_list])]
    # print("no_wd_params: {}".format(no_wd_params_name))
    # res_params = [p for n, p in net.named_parameters() if p.requires_grad and not any([key in n for key in no_wd_list])]
    # optimizer = torch.optim.AdamW(
    #     [
    #         {'params': no_wd_params, 'weight_decay': 0.0},
    #         {'params': res_params}
    #     ],
    #     lr=args.lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=args.decay_rate
    # )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    # scheduler = WarmUpCosineAnnealingLR(
    #     optimizer,
    #     warm_multiplier=10,
    #     warm_duration=5,
    #     cos_duration=args.epoch-5
    # )

    if args.amp:
        scaler = GradScaler()
        logger.log_string("training with fp16 mode.")
    else:
        logger.log_string("training with fp32 mode.")
        scaler = None

    best_psnr = 0.0
    test_dataset = TestSetDataLoader(args, data_name=args.data_name)
    test_loader = DataLoader(dataset=test_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    ''' TRAINING '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))
        net.train()
        if parallel:
            train_loader.sampler.set_epoch(idx_epoch)

        # train epoch
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = \
            train(local_rank, train_loader, device, net, criterion, optimizer, scaler, iters_to_accumulate)

        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))

        # valid model
        if idx_epoch >= 40 and local_rank == 3:  # epoch达到一定轮数后再做test，否则一开始做test没意义
            is_best = False
            psnr_epoch_test, ssim_epoch_test = test(test_loader, device, net)
            print('The %dth Valid, psnr is %.5f, ssim is %.5f' % (idx_epoch + 1, psnr_epoch_test, ssim_epoch_test))
            if psnr_epoch_test > best_psnr:
                is_best = True
                best_psnr = psnr_epoch_test
            if is_best:
                save_ckpt_path = str(
                    checkpoints_dir) + '/%s_%dx%d_%dx_best_model.pth' % (
                        args.model_name, args.angRes, args.angRes,
                        args.scale_factor)
                state = {
                    'epoch':
                    idx_epoch + 1,
                    'state_dict':
                    net.module.state_dict()
                    if hasattr(net, 'module') else net.state_dict(),
                }
                torch.save(state, save_ckpt_path)
                print('Saving the best model at {}, (psnr ssim) {:.3f} {:.3f}'.format(save_ckpt_path, psnr_epoch_test, ssim_epoch_test))
            dist.barrier()
        else:
            dist.barrier()

        # save model at certain interval
        if (idx_epoch + 1) % args.save_epoch == 0:
            if local_rank == 0:
                save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
                    args.model_name, args.angRes, args.angRes, args.scale_factor, idx_epoch + 1)
                state = {
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                }
                torch.save(state, save_ckpt_path)
                logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        ''' scheduler '''
        scheduler.step()
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
        subLFout = torch.zeros(numU, numV, n_colors, args.angRes * args.patch_size_for_test * args.scale_factor,
                               args.angRes * args.patch_size_for_test * args.scale_factor)
        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u:u+1, v:v+1, :, :, :]  # [1,1,3,uh,vw]
                tmp = tmp.squeeze(0)

                hr_coord = make_coord([(tmp.shape[-2] // args.angRes) * args.scale_factor, (tmp.shape[-1] // args.angRes) * args.scale_factor], flatten=False)  # [h',w',2]
                hr_coord = hr_coord.unsqueeze(0)
                hr_coord = hr_coord.to(device)

                cell = torch.ones_like(hr_coord)
                cell[:, 0] *= 2 / ((tmp.shape[-2] // args.angRes) * args.scale_factor)  # 一个cell的高
                cell[:, 1] *= 2 / ((tmp.shape[-1] // args.angRes) * args.scale_factor)  # 一个cell的宽
                cell = cell.to(device)

                out = net(tmp, hr_coord, cell)

                subLFout[u:u+1, v:v+1, :, :, :] = out.squeeze(0)

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
    torch.cuda.empty_cache()
    return psnr_epoch_test, ssim_epoch_test


def train(local_rank, train_loader, device, net, criterion, optimizer, scaler=None, iters_to_accumulate=1):
    '''training one epoch'''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    args.temperature = 1.0
    log_interval = 20
    net.train()

    for idx_iter, data_batch in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):

        data = data_batch['inp'].to(device)      # low resolution
        label = data_batch['gt'].to(device)    # high resolution
        coord = data_batch['coord'].to(device)
        cell = data_batch['cell'].to(device)

        data = (data - 0.5) / 0.5  # 归一化到坐标值域的[-1,1]区间
        label = (label - 0.5) / 0.5  # 归一化到坐标值域的[-1,1]区间

        # 计算loss
        # my_context = net.no_sync if (idx_iter + 1) % iters_to_accumulate != 0 else nullcontext
        # with my_context():
        if scaler is not None:
            with autocast():
                out = net(data, coord, cell)
                loss = criterion(out, label)
        else:
            out = net(data, coord, cell)
            loss = criterion(out, label)

        # 梯度累加
        if iters_to_accumulate > 1:
            loss /= iters_to_accumulate

        # loss反传
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 用梯度更新参数
        if (idx_iter + 1) % iters_to_accumulate == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # 梯度归零
            optimizer.zero_grad()

        loss_iter_train.append(loss.data.cpu())
        psnr, ssim = cal_metrics(args, label, out)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        if idx_iter % log_interval == 0 and local_rank == 0:
            print(f"Train - loss: {loss.item(): .5f}, lr: {optimizer.state_dict()['param_groups'][0]['lr']: .5f} "
                  f"psnr: {psnr: .3f}, ssim: {ssim: .3f}")
        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train


if __name__ == '__main__':
    from option import args

    main(args)
