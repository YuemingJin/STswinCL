import os
import glob
import cv2
import json
import math
import time
import numpy as np
import sys,random
from PIL import Image, ImageOps
sys.path.append('xxx/CaDISv2/')

import torch
import torch.utils.data as data
import torch.nn.functional as F
import albumentations as A

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.cadis_visualization import *
import torchvision.transforms.functional as TF
from torchvision import transforms
MEAN = [0.40789654, 0.44719302, 0.47026115]
STD = [0.28863828, 0.27408164, 0.27809835]

sub_type = {
    '1': remap_experiment1,
    '2': remap_experiment2,
    '3': remap_experiment3
}
sub_class_num = {'1':9, '2':18, '3':26}


train_id = [1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25]
valid_id = [5, 7, 16] #
test_id = [2,12,22] #
Video_IDs = {
    'train': train_id,
    'val': valid_id,
    'test': test_id
}
def load_paths(ids):
    ALL_Paths = dict()
    ALL_Indexes = dict()
    data_dir = 'xxx/CaDISv2/'
    for video_id in ids:
        #print('data path:', data_dir + 'Video{:02d}/Images/*.png'.format(video_id))
        li = glob.glob(data_dir+'Video{:02d}/Images/*.png'.format(video_id))
        li.sort()
        n = len(li)
        ALL_Paths[video_id] = []
        ALL_Indexes[video_id] = []
        for i in range(n):
            ALL_Indexes[video_id].append(i)
            ALL_Paths[video_id].append(li[i])
    return ALL_Indexes, ALL_Paths

class Cata(data.Dataset):
    def __init__(self, split, tag, t=1, arch='swinPlus', global_n=0, downsample=1, step=1):
        super(Cata, self).__init__()
        self.split = split
        self.mean = np.array(MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(STD, dtype=np.float32)[None, None, :]
        self.t = t
        self.tag = tag #?
        self.class_num = sub_class_num[tag]
        self.ids = Video_IDs[self.split] #video ids
        print('!!ID',self.ids)
        self.step = step #frame interval step
        self.remap_experiment = sub_type[tag]
        self.indexes, self.paths = load_paths(self.ids)
        self.arch = arch
        if self.arch == 'swinPlus':
            self.base_size = {'h': 540, 'w': 672} # 'h': 270, 'w': 480
            self.crop_size = {'h': 512, 'w': 640} #'h': 256, 'w': 448
        else:
            self.im_size = {'h': 272, 'w': 480}
#         print(len(self.indexes))
#         print(self.paths)
        
        self.images = []
        for video_id in self.indexes:
            self.images += [[video_id, i] for i in range(len(self.indexes[video_id])) ]
        
        print('EXP {} Loaded {}frames'.format(tag, len(self.images)))
        self.num_samples = len(self.images)
        self.global_n = global_n
        self.downsample = downsample
        print('multi scale input with 512*640')
        
    '''def _loadImg(self, p, padding=True):
        img = cv2.imread(p) #540*960*3
        return img'''

    def _loadMask(self, p, padding=True):
        mask_path = p.replace('Images', 'Labels')
        #print('mask_path:',mask_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY).copy()
        mask, _, _ = self.remap_experiment(mask)
        mask = Image.fromarray(mask.astype('uint8'))
        if self.split == 'train':
            mask = mask.resize((self.base_size['w'], self.base_size['h']), Image.NEAREST)
        return mask

    def _loadImg(self, p, padding=True):
        img = Image.open(p).convert('RGB')
        if self.split == 'train':
            img = img.resize((self.base_size['w'], self.base_size['h']), Image.BILINEAR)
        elif self.split =='val' or 'test':
            img = img.resize((self.crop_size['w'], self.crop_size['h']), Image.BILINEAR)
        return img

    '''def _loadMask(self, p, padding=True):
        mask = Image.open(p)
        return mask'''

    def _random_scale(self, imgs, mask): #训练集的图片和标签做变换base_size_w = 276
        base_size_w = self.base_size['w']
        crop_size_w = self.crop_size['w']
        crop_size_h = self.crop_size['h']
        # random scale (short edge)
        w, h = imgs[0].size
        #print(w,h) #480,270

        long_size = random.randint(int(base_size_w*0.5), int(base_size_w*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else: #here
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        for i in range(len(imgs)):
            imgs[i] = imgs[i].resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        #print(ow,oh) #926,521

        # pad crop
        if short_size < crop_size_w:
            padh = crop_size_h - oh if oh < crop_size_h else 0
            padw = crop_size_w - ow if ow < crop_size_w else 0
            for i in range(len(imgs)):
                imgs[i] = ImageOps.expand(imgs[i], border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size, if has the previous padding above, then do nothing
        w, h = imgs[0].size
        x1 = random.randint(0, w - crop_size_w)
        y1 = random.randint(0, h - crop_size_h)
        for i in range(len(imgs)):
            imgs[i] = np.array(imgs[i].crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        mask = np.array(mask.crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        # final transform
        return imgs, mask

    def load_data(self, video_id, frame_id, split='train', t=1,global_n=0):
        if frame_id > t:
            local_paths = [self.paths[video_id][i] for i in range(frame_id - (t-1)*self.step, frame_id+1,self.step)]
        else:
            local_paths = [self.paths[video_id][i] for i in range(frame_id + (t-1)*self.step, frame_id-1,-self.step)]
#         print(local_paths, video_id, frame_id)
        local_images = [self._loadImg(p) for p in local_paths]
        mask = self._loadMask(local_paths[-1])
        if split == 'train':
            local_images, mask = self._random_scale(local_images, mask)
        else:
            local_images = [np.array(i) for i in local_images]
            mask = np.array(mask)
        return np.array(local_images), np.array(mask)
    
    def transform(self, images, mask):
        # flip
        if np.random.rand()>0.5:
            images = images[:,::-1]
            mask = mask[::-1]
        if np.random.rand()>0.5:
            images = images[:,:,::-1]
            mask = mask[:,::-1]
            
        # noise
        from skimage.util import random_noise
        if np.random.rand()>0.5:
            
            images = [255* random_noise(images[i]/255., mode='gaussian', seed=None, clip=True, var=0.001) for i in range(self.t)]
            images = np.array(images)

        return images.astype('uint8'), mask



    def __getitem__(self, index):
        
        video_id,frame_id = self.images[index]
        if frame_id > self.t:
            local_paths = [self.paths[video_id][i] for i in range(frame_id - (self.t-1)*self.step, frame_id+1,self.step)]
        else:
            local_paths = [self.paths[video_id][i] for i in range(frame_id + (self.t-1)*self.step, frame_id-1,-self.step)]
        #print('local_path:',local_paths)
        frame_id_new = local_paths[-1].split('/')[-1].split('.')[0]
        

        imgs, mask = self.load_data(video_id, frame_id, self.split, self.t, global_n=self.global_n)

        if self.split=='train':
            imgs, mask = self.transform(imgs, mask) #noise&flip
            t, h, w, c = imgs.shape
            images = imgs.transpose((1, 2, 0, 3))
            images = np.ascontiguousarray(images.reshape(h, w, c * t), dtype='uint8')

            #scale = random.random() * 0.4 + 1
            #size = (int(self.im_size['h'] * scale),
            #        int(self.im_size['w'] * scale))
            transf = A.Compose([
                #A.Resize(height=size[0],
                #         width=size[1]),
                #A.RandomCrop(height=self.im_size['h'],
                #             width=self.im_size['w'],
                #             p=1),
                # A.RandomBrightnessContrast(p=0.2),
                A.Rotate(),
                #A.ColorJitter(p=0.5)
            ])

            tsf = transf(image=images, mask=mask)

            images = tsf['image'].reshape(self.crop_size['h'], self.crop_size['w'], t, c) #reshape for dim t
            imgs = np.ascontiguousarray(images.transpose((2, 0, 1, 3)), dtype='float')
            mask = tsf['mask']

        imgs = imgs / 255.
        imgs = (imgs - self.mean) / self.std
        
        if (self.t+self.global_n)==1:
            imgs = imgs[0].transpose(2,0,1) # c w h
        else:
            imgs = imgs.transpose(0,3,1,2) # t c w h
        imgs = torch.from_numpy(imgs)
        
        mask[mask==255] = self.class_num-1 #
        mask = torch.from_numpy(mask.copy())
        mask = F.one_hot(mask.to(torch.int64),num_classes=self.class_num).permute(2,0,1)
        
        return {'path': [video_id,frame_id_new],'image': imgs,'label': mask}

    def __len__(self):
        return self.num_samples



if __name__ == '__main__':
    from tqdm import tqdm
    import pickle

    dataset = Cata('train', tag='2', t=1, downsample=4)
    for d in dataset:
        b1 = d
        print(d['image'].shape)
        print(d['label'].shape)
        break
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                             shuffle=True, num_workers=2,
                                             pin_memory=True, drop_last=True)
