import os,tqdm,sys,time,argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
import torch.cuda.amp as amp
scaler = amp.GradScaler()

import cv2

from net.Ours.base import DeepLabV3Plus

from utils.losses import BCELoss,OhemCELoss2D,DiceLoss
from utils.lr_scheduler import LR_Scheduler_Head
from utils.cata_metrics import segmentation_metrics
from utils.LoadModel import load_model_cata,load_model_full,load_model
from utils.cadis_visualization import *
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.LoadModel import load_model_mswin_CL
from utils.LoadModel import load_model_full

# Training settings
parser = argparse.ArgumentParser(description='real-time segmentation')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='../results/cata')
parser.add_argument('--dataset', type=str, default='cata')
parser.add_argument('--data_tag', type=str, default='type')
parser.add_argument('--log_name', type=str, default='swinPlus_test')#!!
parser.add_argument('--arch', type=str, choices=['deeplab18-nl','puredeeplab18','nlPlus','swinPlus'], default='swinPlus') #!!
parser.add_argument('--pre_log_name', type=str, default=None) #!!
parser.add_argument('--pre_checkpoint', type=str, default=None) #!!

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--loss', type=str, default='ohem')

parser.add_argument('--gpus', type=str, default='1')
parser.add_argument('--downsample', type=int, default=1)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=3)

parser.add_argument('--t', type=int, default=3) ##!!
parser.add_argument('--step', type=int, default=1)

parser.add_argument('--ver', type=int, default=0)
parser.add_argument('--tag', type=int, default=1)

# parser.add_argument('--freeze_name', type=str, )
# parser.add_argument('--spatial_layer', type=int, )
parser.add_argument('--global_n', type=int, default=0)

# parser.add_argument('--pre_name', type=str)
parser.add_argument('--pretrain_ep', type=int, default=20)
parser.add_argument('--decay', type=int, default=2)
# optimizer params
#parser.add_argument('--lr', type=float, default=None, metavar='LR',help='learning rate (default: auto)')
parser.add_argument('--lr-scheduler', type=str, default='poly',help='learning rate scheduler (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9,metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4,metavar='M', help='w-decay (default: 1e-4)')

# parser.add_argument('--need_pretrain', action='store_true')

parser.add_argument('--reset', type=str, default=None)
parser.add_argument('--reset_ep', type=int)


cfg = parser.parse_args()
num_class_table = {'1':8, '2':17, '3':25}

def main():
    # Enviroment
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.gpus
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
    num_gpus = torch.cuda.device_count()
    
    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=num_gpus, rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')
    cfg.log_name += '_ver_'+str(cfg.ver)
    # logger
    cfg.log_dir = os.path.join(cfg.root_dir, 'Tag'+str(cfg.tag),cfg.log_name, 'logs')
    cfg.ckpt_dir = os.path.join(cfg.root_dir, 'Tag'+str(cfg.tag), cfg.log_name, 'ckpt')
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
    print = logger.info
    print(cfg)
    
    # ===============dataset
    print('Setting up data...')
    
    if cfg.dataset=='endovis2018':
        w,h = [512,640]
        from dataset.Endovis2018 import endovis2018
        train_dataset = endovis2018('train', t=cfg.t, rate=1, global_n=cfg.global_n)
        val_dataset = endovis2018('test', t=cfg.t, rate=1, global_n=cfg.global_n)
    elif cfg.dataset=='cata':
        if 'puredeeplab' in cfg.arch:
            w,h = [480,272] #downsample 4 times
        elif 'swin' in cfg.arch:
            w, h = [640, 512]
        ori_w,ori_h = [960,540]
        from dataset.CATA_new_512 import Cata
        train_dataset = Cata('train', t=cfg.t, tag=str(cfg.tag), global_n=cfg.global_n)
        val_dataset = Cata('val', t=cfg.t, tag=str(cfg.tag),  global_n=cfg.global_n)
        classes = train_dataset.class_num

    # ==========build model
    if 'puredeeplab' in cfg.arch:
        model = DeepLabV3Plus(train_dataset.class_num, int(cfg.arch[-2:]))
    elif 'swinPlus' in cfg.arch:
        from net.Ours.base_cata_np import TswinPlusv5
        model = TswinPlusv5(train_dataset.class_num)
    else:
        raise NotImplementedError
    # load pretrain model

    if cfg.pre_log_name:
        cfg.pre_ckpt_path = os.path.join('../CL_output', 'Tag'+str(cfg.tag),cfg.pre_log_name, 'ckpt_epoch_{}.pth'.format(cfg.pre_checkpoint))
        print('initialize the model from:', cfg.pre_ckpt_path)
        model = load_model_mswin_CL(model, cfg.pre_ckpt_path)

    # ==========combile model
    total_params = sum(p.numel() for p in model.parameters())
    print('{0:.2f}m'.format(total_params / 1e6))

    # ==============dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle= True,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                       shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # optimizer using different LR
    params_list = [{'params': model.resnet.parameters(), 'lr': cfg.lr}, ]
    if hasattr(model, 'aspp'):
        params_list.append({'params': model.aspp.parameters(), 'lr': cfg.lr})
    if hasattr(model, 'swin'):
        params_list.append({'params': model.swin.parameters(), 'lr': cfg.lr})
    if hasattr(model, 'project1'):
        params_list.append({'params': model.project1.parameters(), 'lr': cfg.lr})
    if hasattr(model, 'project2'):
        params_list.append({'params': model.project2.parameters(), 'lr': cfg.lr})
    if hasattr(model, 'project3'):
        params_list.append({'params': model.project3.parameters(), 'lr': cfg.lr})
    if hasattr(model, 'classifier'):
        params_list.append({'params': model.classifier.parameters(), 'lr': cfg.lr * 10})


    optimizer = torch.optim.SGD(params_list, lr=cfg.lr,momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    loss_functions = {'bce': BCELoss(), 'ohem':OhemCELoss2D(w*h//16//(cfg.downsample**2)), 'dice':DiceLoss}
    compute_loss = loss_functions[cfg.loss]
    scheduler = LR_Scheduler_Head(cfg.lr_scheduler, cfg.lr, cfg.num_epochs, len(train_loader))
    
    torch.cuda.empty_cache()
    print('Starting training...')
    bestmiou = 0
    bestpac = 0
    bestpa = 0
    best_ep = 0
    
    gpus = cfg.gpus.split(',')
    if len(cfg.gpus)>1:
        model = nn.DataParallel(model, device_ids=list(map(int,gpus))).cuda()
    else:
        model = model.to(cfg.device)


    def train(epoch):
        print('\n Epoch: %d' % epoch)
        model.train()
        tic = time.perf_counter()
        tr_loss = []
        best_pred = 0.0
        for batch_idx, batch in enumerate(train_loader):
            #print('batch_idx',batch_idx)
            for k in batch:
                if not k=='path':
#                     batch[k] = batch[k].to(device=cfg.device, nonw_blocking=True).float()
                    batch[k] = batch[k].to(device=cfg.device).float()

            scheduler(optimizer, batch_idx, epoch, best_pred)
            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(batch['image']) #4,12,512,640

                if cfg.loss == 'ohem':
                    loss = compute_loss(outputs, torch.argmax(batch['label'],dim=1).long())
                else:
                    loss = compute_loss(outputs, batch['label'])
            tr_loss.append(loss.detach().cpu().numpy())

            #loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                lr = optimizer.param_groups[0]['lr']
                print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                     'loss:{:.4f} Time:{:.4f} lr:{:.7f}'.format(loss.item(),duration,lr))



        summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch)
        return
    
    def val_map(epoch):
        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()
        gt = []
        prediction = []

        with torch.no_grad():
            for inputs in tqdm.tqdm(val_loader):
                inputs['image'] = inputs['image'].to(cfg.device).float()
#                 tmp = torch.argmax(inputs['label'],dim=1).numpy()
#                 print(np.unique(tmp))
#                 continue
                tic = time.perf_counter()
                output = model(inputs['image'])
                output = F.interpolate(output, (ori_h, ori_w), mode='bilinear', align_corners=True)
                output = F.softmax(output,dim=1)
                output = torch.argmax(output,dim=1)
                output = output.cpu().numpy()
                duration = time.perf_counter() - tic

                gt_i = torch.argmax(inputs['label'], dim=1).numpy()
                gt.append(gt_i[0])  # as batch size = 1, therefore [0]
                # print(inputs['label'].shape)
                prediction.append(output[0])

            pa, pac, pac_c, miou, miou_c = segmentation_metrics(gt, prediction,
                                                                num_classes=num_class_table[str(cfg.tag)])

            pac_c = [float('{:.4f}'.format(i)) for i in pac_c]
            miou_c = [float('{:.4f}'.format(i)) for i in miou_c]
            print('PA:{:.4f} PAC:{:.4f} mIoU:{:.4f} Time:{:.4f}'.format(pa, pac, miou, duration))
            print('Class PA:{}'.format(' '.join(map(str, pac_c))))
            print('Class IoU:{}'.format(' '.join(map(str, miou_c))))

            summary_writer.add_scalar('PA', pa, epoch)
            summary_writer.add_scalar('PAC', pac, epoch)
            summary_writer.add_scalar('mIoU', miou, epoch)

            return miou

    if cfg.reset:
        pre_ckpt_path = os.path.join(cfg.root_dir, 'Tag'+str(cfg.tag), cfg.log_name, 'ckpt', 'latestcheckpoint.t7'.format(cfg.reset_ep))
        print('initialize the model from:',pre_ckpt_path)
        model = load_model_full(model,pre_ckpt_path)
        best_ep = cfg.reset_ep
        bestmiou = val_map(best_ep)
        
    for epoch in range(best_ep+1, cfg.num_epochs + 1):
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            save_iou = val_map(epoch)
            if save_iou>bestmiou:
                bestmiou = save_iou
                best_ep = epoch
                print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch)))
                print(saver.save(model.state_dict(), 'bestcheckpoint'))
            else:
                if epoch-best_ep>200:
                    break
            print(saver.save(model.state_dict(), 'latestcheckpoint'))
    summary_writer.close()
        
if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
