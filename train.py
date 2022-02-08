from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader
from collections import OrderedDict

import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

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
    if args.use_pre_pth == False:
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
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    # net = net.to(device)
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
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    if args.amp:
        scaler = GradScaler()
        logger.log_string("training with fp16 mode.")
    else:
        logger.log_string("training with fp32 mode.")
        scaler = None

    ''' TRAINING '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))
        if parallel:
            train_loader.sampler.set_epoch(idx_epoch)

        # train epoch
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = \
            train(local_rank, train_loader, device, net, criterion, optimizer, scaler, iters_to_accumulate)

        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))

        # save model
        if args.local_rank == 0:
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


def train(local_rank, train_loader, device, net, criterion, optimizer, scaler=None, iters_to_accumulate=1):
    '''training one epoch'''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    args.temperature = 1.0
    log_interval = 20
    net.train()

    for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        data = data.to(device)      # low resolution
        label = label.to(device)    # high resolution
        # out = net(data)
        # 计算loss
        if scaler is not None:
            with autocast():
                out = net(data)
                loss = criterion(out, label)
        else:
            out = net(data)
            loss = criterion(out, label)

        # 梯度累加
        if iters_to_accumulate != 1:
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
        # torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())
        psnr, ssim = cal_metrics(args, label, out)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        if idx_iter % log_interval == 0 and local_rank == 0:
            print(f"Train - loss: {loss.item():.5f}, ssim: {psnr:.5f}, psnr: {ssim:.5f}")

        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train


if __name__ == '__main__':
    from option import args

    main(args)
