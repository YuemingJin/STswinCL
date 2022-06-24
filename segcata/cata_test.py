import os,tqdm,sys,time,argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist


from utils.losses import BCELoss,OhemCELoss2D
from utils.cata_metrics import segmentation_metrics
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.LoadModel import load_model_test
from utils.cadis_visualization_test import *
from skimage import io

# Training settings
parser = argparse.ArgumentParser(description='real-time segmentation')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')
parser.add_argument('--root_dir', type=str, default='../results/cata') #!!
parser.add_argument('--log_name', type=str, default='basecata') #!!
parser.add_argument('--checkpoint', type=str,default='200')
parser.add_argument('--arch', type=str, choices=['puredeeplab18','swinPlus'], default='puredeeplab18')

parser.add_argument('--tag', type=int, default=2) #!!
parser.add_argument('--ver',type=int, default = 0)
parser.add_argument('--layer', type=int, default=18)
parser.add_argument('--load_model', type=str, default=None)

parser.add_argument('--gpus', type=str, default='3')
parser.add_argument('--downsample', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=3)

parser.add_argument('--t', type=int, default=1)
parser.add_argument('--swin_h', type=int, default=32)
parser.add_argument('--swin_w', type=int, default=56)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--global_n', type=int, default=0)
cfg = parser.parse_args()
num_class_table = {1:8, 2:17, 3:25} #? -1 compared with data loader

test_id = [2, 12, 22]

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
        
    # logger
    cfg.log_dir = os.path.join(cfg.root_dir, 'Tag'+str(cfg.tag),cfg.log_name, 'logs_test_final')
    cfg.ckpt_dir = os.path.join(cfg.root_dir, 'Tag'+str(cfg.tag),cfg.log_name, 'ckpt')
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    for k in test_id:
        cfg.vis_dir = os.path.join(cfg.log_dir, 'visualization','Video_'+str(k))
        os.makedirs(cfg.vis_dir, exist_ok=True)
    cfg.vis_path = os.path.join(cfg.log_dir, 'visualization','Video_{}/{}.png')
       
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    print = logger.info
    print(cfg)
    
    # dataset
    print('Setting up data...')
    from dataset.CATA_new_512 import Cata
    if 'puredeeplab' in cfg.arch:
        w, h = [480, 272]
    elif 'swin' in cfg.arch:
        w, h = [640, 512]
    ori_w, ori_h = [960, 540]
    test_dataset = Cata('test', t=cfg.t, arch=cfg.arch, tag=str(cfg.tag), global_n=cfg.global_n)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                       shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # build model
    if 'puredeeplab' in cfg.arch:
        from net.Ours.base import DeepLabV3Plus #no resnet pretrain
        model = DeepLabV3Plus(test_dataset.class_num, cfg.layer)
    elif 'swin' in cfg.arch:
        from net.Ours.base_cata_np import TswinPlusv5
        model = TswinPlusv5(num_classes=test_dataset.class_num)

    else:
        raise NotImplementedError
    
    # combile model
    torch.cuda.empty_cache()
    gpus = cfg.gpus.split(',')
    if len(cfg.gpus)>1:
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    else:
        model = model.to(cfg.device)
        

    cfg.load_model = os.path.join(cfg.ckpt_dir, 'epoch_{}_checkpoint.t7'.format(cfg.checkpoint)) #!!
    print('model path:', cfg.load_model)
    model = load_model_test(model, cfg.load_model)
    
    print('=> Begin to test ......')
    def val_map(epoch):
        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()
        gt = []
        prediction = []
        metrics = np.zeros((2,))
        count = 0
        with torch.no_grad():
            for inputs in tqdm.tqdm(test_loader):
                inputs['image'] = inputs['image'].to(cfg.device).float()

                tic = time.perf_counter()
                output_ori = model(inputs['image'])
                output = F.interpolate(output_ori, (ori_h, ori_w), mode="bilinear")

                output = F.softmax(output,dim=1)
                output = torch.argmax(output,dim=1) #bs,h,w
                output = output.cpu().numpy()

                duration = time.perf_counter() - tic

                # # # #=====visualize figure======
                # predict = output.astype(np.uint8)
                # predict = predict.squeeze(0)
                # #print('predict:',predict.shape)
                #
                # ins = int(inputs['path'][0])
                # i = inputs['path'][1][0]
                # save_pth = cfg.vis_path.format(ins, i)
                # ## ------- save image with GT mask --------
                # image = inputs['image']
                # image = image[:,-1,:,:,:]
                # #print(image.shape)
                # image = image.permute((0,2,3,1)).cpu().numpy()
                # image = image.squeeze(0)
                #
                # fig = plot_experiment_foreval(image,predict,cfg.tag)
                # io.imsave(save_pth, fig)
                # # # ## --------------------------

                gt_i = torch.argmax(inputs['label'],dim=1).numpy()
                gt.append(gt_i[0])

                prediction.append(output[0])


            pa, pac, pac_c, miou, miou_c = segmentation_metrics(gt,prediction,num_classes=num_class_table[cfg.tag])


        pac_c = [float('{:.4f}'.format(i)) for i in pac_c]
        miou_c = [float('{:.4f}'.format(i)) for i in miou_c]
        print('PA:{:.4f} PAC:{:.4f} mIoU:{:.4f}'.format(pa, pac, miou))
        print('Class PA:{}'.format(' '.join(map(str,pac_c))))
        print('Class IoU:{}'.format(' '.join(map(str, miou_c))))
        return miou
    val_map(0)
    
if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
