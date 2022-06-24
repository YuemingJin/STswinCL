import os,tqdm,sys,time,argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np
import torch.cuda.amp as amp
scaler = amp.GradScaler()

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
from net.Ours.base import DeepLabV3,DeepLabV3Plus

from utils.losses import BCELoss,OhemCELoss2D,DiceLoss

from utils.cata_metrics import segmentation_metrics
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.LoadModel import load_model_cata,load_model_full,load_model
from utils.cadis_visualization import *

# Training settings
parser = argparse.ArgumentParser(description='real-time segmentation')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='../results/cata')
parser.add_argument('--dataset', type=str, default='cata')
parser.add_argument('--data_tag', type=str, default='type')
parser.add_argument('--log_name', type=str, default='cata_test')
parser.add_argument('--arch', type=str, choices=['puredeeplab18','swinPlus'], default='swinPlus')
parser.add_argument('--pre_log_name', type=str, default=None)
parser.add_argument('--pre_checkpoint', type=str, default=None) #!!

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--loss', type=str, default='ohem')

parser.add_argument('--gpus', type=str, default='3')
parser.add_argument('--downsample', type=int, default=1)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=3)

parser.add_argument('--t', type=int, default=1)
parser.add_argument('--step', type=int, default=1)

parser.add_argument('--ver', type=int, default=1)
parser.add_argument('--tag', type=int, default=1)


# parser.add_argument('--freeze_name', type=str, )
# parser.add_argument('--spatial_layer', type=int, )
parser.add_argument('--global_n', type=int, default=0)

parser.add_argument('--pretrain_ep', type=int, default=20)
parser.add_argument('--decay', type=int, default=2)

parser.add_argument('--reset', type=str, default=None)
parser.add_argument('--reset_ep', type=int)


cfg = parser.parse_args()
num_class_table = {'1':8, '2':17, '3':25} #? -1 compared with data loader

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

    cfg.log_name += '_ver_' + str(cfg.ver)

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
        classes = train_dataset.class_num
    elif cfg.dataset=='cata':
        if 'puredeeplab' in cfg.arch:
            w,h = [480,272] #downsample 4 times
        elif 'swin' in cfg.arch:
            w, h = [640, 512]
        ori_w,ori_h = [960,540]
        from dataset.CATA_new_512 import Cata
        train_dataset = Cata('train', t=cfg.t, tag=str(cfg.tag), arch=cfg.arch, global_n=cfg.global_n)
        val_dataset = Cata('val', t=cfg.t, tag=str(cfg.tag), arch=cfg.arch, global_n=cfg.global_n)
        classes = train_dataset.class_num

    # ==========build model
    if 'puredeeplab' in cfg.arch:
        model = DeepLabV3Plus(train_dataset.class_num, 18)
    elif 'swin' in cfg.arch:
        from net.Ours.base_cata_np import TswinPlusv5
        model = TswinPlusv5(train_dataset.class_num)
    else:
        raise NotImplementedError
    # load pretrain model
    if cfg.pre_log_name is not None:
        cfg.pre_ckpt_path = os.path.join(cfg.root_dir, 'Tag'+str(cfg.tag), cfg.pre_log_name, 'ckpt', 'epoch_{}_checkpoint.t7'.format(cfg.pre_checkpoint))
        print('initialize the model from:', cfg.pre_ckpt_path)
        model = load_model(model, cfg.pre_ckpt_path)
    
    # ==========combile model
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    loss_functions = {'bce': BCELoss(), 'ohem':OhemCELoss2D(w*h//16//(cfg.downsample**2),ignore_index=int(num_class_table[str(cfg.tag)])), 'dice':DiceLoss}
    compute_loss = loss_functions[cfg.loss] 
    
    torch.cuda.empty_cache()
    print('Starting training...')
    best = 0
    best_ep = 0
    
    gpus = cfg.gpus.split(',')
    if len(cfg.gpus)>1:
        model = nn.DataParallel(model, device_ids=list(map(int,gpus))).cuda()
    else:
        model = model.to(cfg.device)

    # ==============dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle= True,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                       shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

#     return
    def train(epoch):
        print('\n Epoch: %d' % epoch)
        model.train()
        tic = time.perf_counter()
        tr_loss = []
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                if not k=='path':
#                     batch[k] = batch[k].to(device=cfg.device, nonw_blocking=True).float()
                    batch[k] = batch[k].to(device=cfg.device).float()

            with amp.autocast():
                outputs = model(batch['image'])

                if cfg.loss == 'ohem':
                    loss = compute_loss(outputs, torch.argmax(batch['label'],dim=1).long())
                else:
                    loss = compute_loss(outputs, batch['label'])
            tr_loss.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                     'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
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
                tic = time.perf_counter()
                output = model(inputs['image'])

                output = F.interpolate(output, (ori_h,ori_w), mode='bilinear', align_corners=True)
                output = F.softmax(output,dim=1)
                output = torch.argmax(output,dim=1)
                output = output.cpu().numpy()
                duration = time.perf_counter() - tic

                ## official eval
                gt_i = torch.argmax(inputs['label'],dim=1).numpy()
                gt.append(gt_i[0])
                prediction.append(output[0])

            pa, pac, pac_c, miou, miou_c = segmentation_metrics(gt, prediction, num_classes=num_class_table[str(cfg.tag)])


        pac_c = [float('{:.4f}'.format(i)) for i in pac_c]
        miou_c = [float('{:.4f}'.format(i)) for i in miou_c]
        print('PA:{:.4f} PAC:{:.4f} mIoU:{:.4f} Time:{:.4f}'.format(pa, pac, miou, duration))
        print('Class PA:{}'.format(' '.join(map(str,pac_c))))
        print('Class IoU:{}'.format(' '.join(map(str, miou_c))))

        summary_writer.add_scalar('PA', pa, epoch)
        summary_writer.add_scalar('PAC', pac, epoch)
        summary_writer.add_scalar('mIoU', miou, epoch)

        return miou

    if cfg.reset:
        pre_ckpt_path = os.path.join(cfg.root_dir, 'Tag'+str(cfg.tag), cfg.log_name,'ckpt', 'epoch_{}_checkpoint.t7'.format(cfg.reset_ep))
        print('initialize the model from:',pre_ckpt_path)
        model = load_model_full(model,pre_ckpt_path)
        best_ep = cfg.reset_ep
        best = val_map(best_ep)
        
    for epoch in range(best_ep+1, cfg.num_epochs + 1):
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            save_map = val_map(epoch)
            if save_map>best:
                best = save_map
                best_ep = epoch
                print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch)))
            else:
                if epoch-best_ep>200:
                    break
            print(saver.save(model.state_dict(), 'latestcheckpoint'))
    summary_writer.close()
        
if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
