import json
import os
import time
from shutil import copyfile

import torch
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# from contrast.models.PixPro import PixPro
from contrast.models.PixPro_swin_v5 import ConsistencyLoss
# from contrast import resnet
from contrast.data import get_loader
from contrast.logger import setup_logger
from contrast.lr_scheduler import get_scheduler
from contrast.option import parse_option
from contrast.util import AverageMeter
from contrast.lars import add_weight_decay, LARS




import torch.cuda.amp as amp
scaler = amp.GradScaler()

def build_model(args):
    # encoder = resnet.__dict__[args.arch]
    # model = models.__dict__[args.model](encoder, args).cuda()
    model = ConsistencyLoss(args).cuda()
    # print(model.parameters())

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.pixpro.parameters(),
            lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,)
    elif args.optimizer == 'lars':
        params = add_weight_decay(model.pixpro, args.weight_decay)
        optimizer = torch.optim.SGD(
            params,
            lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate,
            momentum=args.momentum,)
        optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    return model, optimizer


def load_pretrained(model, pretrained_model):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    state_dict = ckpt['model']
    model_dict = model.state_dict()

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    logger.info(f"==> loaded checkpoint '{pretrained_model}' (epoch {ckpt['epoch']})")


def load_checkpoint(args, model, optimizer, scheduler, sampler=None):
    logger.info(f"=> loading checkpoint '{args.resume}'")

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.module.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info(f"=> loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, sampler=None):
    logger.info('==> Saving...')
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    state = {
        'opt': args,
        'model': model.state_dict(),
        # 'encoder': model.pixpro.encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))


def main(args):
    train_prefix = 'train'
    train_loader = get_loader(args) #data loader

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    model, optimizer = build_model(args)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    # optionally resume from a checkpoint
    if args.pretrained_model:
        assert os.path.isfile(args.pretrained_model)
        load_pretrained(model, args.pretrained_model)
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, optimizer, scheduler, sampler=train_loader.sampler)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    for epoch in range(args.start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train(epoch, train_loader, model, optimizer, scheduler, args, summary_writer)
        # fake_train(train_loader)

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save_checkpoint(args, epoch, model, optimizer, scheduler, sampler=train_loader.sampler)

    

def train(epoch, train_loader, model, optimizer, scheduler, args, summary_writer):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data = [item.cuda(non_blocking=True) for item in data]

        optimizer.zero_grad()


        with amp.autocast():
            loss = model(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9],data[10], data[11])
            # directly output loss

        ## update part
        scaler.scale(loss).backward()
        # l = [x.grad.mean() for x in optimizer.param_groups[0]['params']]
        # print(sum(l) / len(l))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # scheduler.step()

        # update meters and print info
        loss_meter.update(loss.item(), data[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        train_len = len(train_loader)
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'lr {lr:.3f}  '
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')

            # tensorboard logger
            if summary_writer is not None:
                step = (epoch - 1) * len(train_loader) + idx
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', loss_meter.val, step)

def fake_train(train_loader):
    for idx, data in enumerate(train_loader):
        H, W = (28, 28)
        pos_ratio = 0.7
        coord_q = data[2]
        coord_k = data[2]
        print(coord_q, coord_k)
        x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
        y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)
        # [bs, 1, 1]
        q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
        q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
        k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
        k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
        # [bs, 1, 1]
        q_start_x = coord_q[:, 0].view(-1, 1, 1)
        q_start_y = coord_q[:, 1].view(-1, 1, 1)
        k_start_x = coord_k[:, 0].view(-1, 1, 1)
        k_start_y = coord_k[:, 1].view(-1, 1, 1)

        # [bs, 1, 1]
        q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
        k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
        max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

        # [bs, 7, 7]
        center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (y_array + 0.5) * k_bin_height + k_start_y # project to the coordinate in the original image

        # [bs, 49, 49]
        dist_center = torch.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                                + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
        pos_mask = (dist_center < pos_ratio).float().detach().numpy()
        pos_mask = pos_mask[0].reshape(-1)
        print(sum(pos_mask==0), sum(pos_mask==1))
        break


if __name__ == '__main__':
    opt = parse_option(stage='pre-train')

    if opt.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(opt)).items()))
    )

    main(opt)
