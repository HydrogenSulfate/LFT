import importlib
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import LFdivide, LFintegrate, Logger, cal_metrics, create_dir
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
    else:
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
                logger.log_string('resume training and load [1.model] state dict')
            except Exception as e:
                logger.log_string(str(e))
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('resume training and load [1.model] state dict')
        except Exception as e:
            logger.log_string(str(e))
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
        # pass

    local_rank = args.local_rank
    net = net.to(local_rank)
    if parallel:
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
    if args.use_pre_pth:
        logger.log_string('resume training and load [2.optimizer] state dict')
        optimizer.load_state_dict(checkpoint['opt'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    if args.use_pre_pth:
        logger.log_string("resume training and load [3.scheduler] state dict")
        scheduler.load_state_dict(checkpoint['lr'])

    if args.amp:
        scaler = GradScaler()
        if args.use_pre_pth:
            scaler.load_state_dict(checkpoint['scaler'])
            logger.log_string("resume training and load [4.scaler] state dict")
        logger.log_string("training with fp16 mode.")
    else:
        logger.log_string("training with fp32 mode.")
        scaler = None

    best_psnr = 0.0
    test_dataset = TestSetDataLoader(args, data_name=args.test_data_name)
    test_loader = DataLoader(dataset=test_dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    ''' TRAINING '''
    if args.use_pre_pth:
        del checkpoint
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\n[Train] [Epoch %d /%s]:' % (idx_epoch + 1, args.epoch))
        net.train()
        if parallel:
            train_loader.sampler.set_epoch(idx_epoch)

        # train epoch
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = \
            train_epoch(local_rank, train_loader, device, net, criterion, optimizer, scaler, iters_to_accumulate)

        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))

        # save model at certain interval
        if (idx_epoch + 1) % args.save_epoch == 0:
            if local_rank == 0:
                save_ckpt_path = str(checkpoints_dir) + '/%s_%d_%d_epoch_%02d_model.pth' % (
                    args.model_name, args.angRes_in, args.angRes_out, idx_epoch + 1)
                state = {
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                    'opt': optimizer.state_dict(),
                    'lr': scheduler.state_dict(),
                }
                if args.amp:
                    state.update({'scaler': scaler.state_dict()})
                torch.save(state, save_ckpt_path)
                logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        # valid model
        if local_rank == 0:  # epoch达到一定轮数后再做test，否则一开始做test没意义
            is_best = False
            psnr_epoch_test, ssim_epoch_test = test(test_loader, device, net)
            print('The %dth Valid, psnr is %.5f, ssim is %.5f' % (idx_epoch + 1, psnr_epoch_test, ssim_epoch_test))
            if psnr_epoch_test > best_psnr:
                is_best = True
                best_psnr = psnr_epoch_test
            if is_best:
                save_ckpt_path = str(
                    checkpoints_dir) + '/%s_%dx%d_best_model.pth' % (args.model_name, args.angRes_in, args.angRes_out)
                state = {
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                    'opt': optimizer.state_dict(),
                    'lr': scheduler.state_dict()
                }
                if args.amp:
                    state.update({'scaler': scaler.state_dict()})
                torch.save(state, save_ckpt_path)
                print('Saving the best model at {}, (psnr ssim) {:.3f} {:.3f}'.format(save_ckpt_path, psnr_epoch_test, ssim_epoch_test))
            if parallel:
                dist.barrier()
        else:
            if parallel:
                dist.barrier()

        ''' scheduler '''
        scheduler.step()
        pass
    pass


@torch.no_grad()
def test(test_loader: DataLoader, device: torch.device, net: torch.nn.Module) -> Tuple[float, float]:
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args).cuda()
    net.eval()

    psnr_iter_test = []
    ssim_iter_test = []

    for idx_iter, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):

        pre: torch.Tensor = data_batch['pre'].to(device)      # low resolution
        nxt: torch.Tensor = data_batch['nxt'].to(device)      # low resolution
        label: torch.Tensor = data_batch['gt'].to(device)      # high resolution

        assert label.shape[0] == 1 and label.ndim == 4

        N, n_colors, uh, vw = label.shape

        h0, w0 = int(uh//args.angRes_out), int(vw//args.angRes_out)

        subLFin_pre = LFdivide(pre, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        subLFin_nxt = LFdivide(nxt, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
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
                print(f"input.shape: {tmp_pre.shape}, output.shape: {out.shape}")

                subLFout[u:u+1, v:v+1, :, :, :] = out
                print(f"inference patch: {u*numV+v}/{numU*numV}")
        Sr_4D_rgb = LFintegrate(
            subLFout,
            args.angRes_out,
            args.patch_size_for_test,
            args.stride_for_test,
            h0,
            w0
        )  # [u,v,3,h,w]
        Sr_SAI_rgb = Sr_4D_rgb.permute(2, 0, 3, 1, 4).reshape(
            n_colors,
            h0 * args.angRes_out,
            w0 * args.angRes_out
        )  # [3, uh,vw]
        psnr, ssim = cal_metrics(args.angRes_out, label, Sr_SAI_rgb)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    torch.cuda.empty_cache()
    return psnr_epoch_test, ssim_epoch_test


def train_epoch(local_rank, train_loader, device, net, criterion, optimizer, scaler=None, iters_to_accumulate=1):
    '''training one epoch'''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    args.temperature = 1.0
    log_interval = 20
    net.train()

    for idx_iter, data_batch in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):

        pre: torch.Tensor = data_batch['pre'].to(device)      # low resolution
        next: torch.Tensor = data_batch['nxt'].to(device)      # low resolution
        label: torch.Tensor = data_batch['gt'].to(device)      # high resolution

        # 计算loss
        # my_context = net.no_sync if (idx_iter + 1) % iters_to_accumulate != 0 else nullcontext
        # with my_context():
        if scaler is not None:
            with autocast():
                out = net(pre, next)
                loss = criterion(out, label)
        else:
            out = net(pre, next)
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

        loss_iter_train.append(loss.item())
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
    # test(None, None, None)
