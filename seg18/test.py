import os,tqdm,sys,time,argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist

from utils.losses import BCELoss,OhemCELoss2D
from utils.EndoMetric import general_dice, general_jaccard
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.LoadModel import load_model_full_fortest
from skimage import io

# Training settings
parser = argparse.ArgumentParser(description='real-time segmentation')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')
parser.add_argument('--root_dir', type=str, default='../results/endo18')
parser.add_argument('--dataset', type=str, default='endovis2018')
parser.add_argument('--data_tag', type=str, default='type')
parser.add_argument('--log_name', type=str, default='swinPlusv2_ver_9')
parser.add_argument('--checkpoint', type=str,default='9')
parser.add_argument('--layer', type=int, default=18)
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--arch', type=str, choices=['deeplab18-nl','puredeeplab18','nlPlus','swinPlus'], default='swinPlus') #!!


parser.add_argument('--gpus', type=str, default='3')
parser.add_argument('--downsample', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=3)
parser.add_argument('--test_bs', type=int, default=1)
parser.add_argument('--t', type=int, default=3)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--global_n', type=int, default=0)
cfg = parser.parse_args()


#bg = black
color_map = {
    0: [0,0,0], # background-tissue
    1: [0,255,0], # instrument-shaft
    2: [0,255,255], # instrument-clasper
    3: [125,255,12], # instrument-wrist
    4: [255,55,0], # kidney-parenchyma，
    5: [24,55,125], # covered-kidney，
    6: [187,155,25], # thread，
    7: [0,255,125], # clamps，
    8: [255,255,125], # suturing-needle
    9: [123,15,175], # suction-instrument，
    10: [124,155,5], # small-intestine
    11: [12,255,141] # ultrasound-probe,
}


def label2rgb(ind_im, color_map=color_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im


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
    cfg.log_dir = os.path.join(cfg.root_dir, cfg.log_name, 'logs_test_time')
    os.makedirs(cfg.log_dir, exist_ok=True)
    cfg.ckpt_dir = os.path.join(cfg.root_dir, cfg.log_name, 'ckpt')

    for k in range(1,5):
        cfg.vis_dir = os.path.join(cfg.log_dir, 'visualization_'+str(cfg.checkpoint),'seq_'+str(k))
        os.makedirs(cfg.vis_dir, exist_ok=True)
    cfg.vis_path = os.path.join(cfg.log_dir, 'visualization_'+str(cfg.checkpoint), 'seq_{}/frame{:03d}.png')
       
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    print = logger.info
    print(cfg)
    
    # dataset
    print('Setting up data...')
    

    h,w = [512,640]
    ori_h, ori_w = [1024, 1280]
    from dataset.Endovis2018_new import endovis2018
    test_dataset = endovis2018('test', t=cfg.t, rate=1, global_n=cfg.global_n)
    classes = test_dataset.class_num
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                       shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # build model
    if 'puredeeplab' in cfg.arch:
        from net.Ours.base_cata_np import DeepLabV3Plus
        model = DeepLabV3Plus(test_dataset.class_num, int(cfg.arch[-2:]))
    elif 'swinPlus' in cfg.arch:
        from net.Ours.base18 import TswinPlus
        model = TswinPlus(test_dataset.class_num)
    else:
        raise NotImplementedError
    
    # combile model
    
    torch.cuda.empty_cache()
    
    gpus = cfg.gpus.split(',')
    if len(cfg.gpus)>1:
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    else:
        model = model.to(cfg.device)
        
    if cfg.load_model is None:
        cfg.load_model = os.path.join(cfg.ckpt_dir, 'epoch_{}_checkpoint.t7'.format(cfg.checkpoint))
        print('model path:', cfg.load_model)
        
    model = load_model_full_fortest(model, cfg.load_model)
    
    def val_map(epoch):
        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()
        metrics = np.zeros((2,))
        metrics_seq = np.zeros((2, 4))
        count_seq = np.zeros((4,))
        dice_each = np.zeros((12,))
        iou_each = np.zeros((12,))
        tool_eac = np.zeros((12,))
        count = 0

        with torch.no_grad():
            for inputs in tqdm.tqdm(test_loader):
                inputs['image'] = inputs['image'].to(cfg.device).float()
                #print('shape:', inputs['image'].shape) #1,3,256,480
                tic = time.perf_counter()
                output = model(inputs['image'])

                output = F.interpolate(output, (ori_h,ori_w), mode='bilinear', align_corners=True)
                output = F.softmax(output,dim=1)
                output = torch.argmax(output,dim=1)
                output = output.cpu().numpy()
                duration = time.perf_counter() - tic
                print('duration:',duration)

                # #=====visualize figure======
                # predict = output.astype(np.uint8)
                # ins = int(inputs['path'][0])
                # i = int(inputs['path'][1])
                # save_pth = cfg.vis_path.format(ins, i)
                # # print('input path:', save_pth)
                # predict = label2rgb(predict[0])
                # io.imsave(save_pth, predict)

                dice = general_dice(torch.argmax(inputs['label'], dim=1).numpy(),output)  # dice containing each tool class
                iou = general_jaccard(torch.argmax(inputs['label'], dim=1).numpy(), output)

                for i in range(len(dice)):
                    tool_id = dice[i][0]
                    dice_each[tool_id] += dice[i][1]
                    iou_each[tool_id] += iou[i][1]
                    tool_eac[tool_id] += 1

                frame_dice = np.mean([dice[i][1] for i in range(len(dice))])
                frame_iou = np.mean([iou[i][1] for i in range(len(dice))])
                #overall
                metrics[0] += frame_dice # dice of each frame
                metrics[1] += frame_iou
                count += 1

                #----for seq
                seq_ind = int(inputs['path'][0]) - 1 #seq: 0-3
                metrics_seq[0][seq_ind] += frame_dice
                metrics_seq[1][seq_ind] += frame_iou
                count_seq[seq_ind] += 1


            print(count)
            metrics[0] /= count
            metrics[1] /= count
            print(metrics)
            dc, jc = metrics[0], metrics[1]

            metrics_seq[0] /= count_seq
            dice_seq = [float('{:.4f}'.format(i)) for i in metrics_seq[0]]
            metrics_seq[1] /= count_seq
            iou_seq = [float('{:.4f}'.format(i)) for i in metrics_seq[1]]

        print('Dice:{:.4f} IoU:{:.4f} Time:{:.4f}'.format(dc, jc, duration))
        print('Dice_seq1:{:.4f}, seq2:{:.4f}, seq3:{:.4f}, seq4:{:.4f}'.format(dice_seq[0], dice_seq[1], dice_seq[2],dice_seq[3]))
        print('IOU_seq1:{:.4f}, seq2:{:.4f}, seq3:{:.4f}, seq4:{:.4f}'.format(iou_seq[0], iou_seq[1], iou_seq[2],iou_seq[3]))
        return jc
    val_map(0)
    
if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
