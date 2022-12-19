from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import one_hot 
from tqdm import tqdm
from PIL import Image, ImageOps

import albumentations as A
import numpy as np
import torch
import random
import json
import cv2
import os

LABEL_JSON = '../2018data/ead2018/train/labels.json'
DATA_ROOT = '../2018data/ead2018/'

Procedures = {'train':[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]}

class endovis2018(Dataset):

    def __init__(self, split, t=1, arch='swinPlus', rate=1, global_n=0):
        super(endovis2018, self).__init__()

        #self.im_size = {'h': 512, 'w': 640}
        self.class_num = 12
#         self.class_num = 11

        self.mode = split
        self.t = t
        self.global_n = global_n
        self.rate = rate
        self.arch = arch
        if self.arch == 'swinPlus':
            self.base_size = {'h': 540, 'w': 672} 
            self.crop_size = {'h': 512, 'w': 640} 
        else:
            self.im_size = {'h': 512, 'w': 640}

        if self.mode == 'train':
            train_images = [[f,i] for i in range(149) for f in Procedures['train']]
            self.images = train_images

        self.test = (split =='test')
        if self.test:
            test_images = [[s, i] for s in range(1, 2) for i in range(250)] #250
            test_images_2 = [[s, i] for s in range(2,5) for i in range(249)] #249
            self.images = test_images +test_images_2

        self.num_samples = len(self.images)
        print(f'Loaded {self.num_samples} frames. For {self.mode}')

        with open(LABEL_JSON, 'r') as f:
            self.lb_json = json.load(f)
        self.json_color = [item['color'] for item in self.lb_json]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ins, frame = self.images[idx]
        images, label = self._load_data(ins, frame, self.t, self.global_n)
#         print(np.unique(label), images[0].shape, label.shape)
        images = np.array(images).astype('uint8')


        ###=========augmentation===============
        if self.mode == 'train':
            t, h, w, c = images.shape
            images = images.transpose((1,2,0,3))
            images = np.ascontiguousarray(images.reshape(h,w,c*t), dtype='uint8')

            transf = A.Compose([
#                 A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                # A.HueSaturationValue(p=0.5),
                A.Rotate()
                # A.ColorJitter(p=0.5)
            ])
            
            tsf = transf(image=images, mask=label)
            
            images = tsf['image'].reshape(self.crop_size['h'], self.crop_size['w'],t,c)
            images =  np.ascontiguousarray(images.transpose((2,0,1,3)), dtype='float')
            label = tsf['mask']
#             print(images.shape, np.unique(label))



        #==========image and label=========
        images = images.astype('float')
        images /= 255.
        if self.t + self.global_n == 1:
            images = images[0].transpose(2,0,1)     # c * w * h
        else:
            images = images.transpose(0,3,1,2)  # t * c * w * h
        images = torch.from_numpy(images)

        h, w = label.shape[:2]
        label = label[::self.rate, ::self.rate]
        label = torch.from_numpy(label)
        
        if self.class_num==11:  ###delete the class that test data does not have
            label[label==9] = 0
            label[label>9] -= 1

        label = one_hot(label.to(torch.int64), num_classes=self.class_num)
        label = label.permute(2,0,1) 

        return {'path': [ins, frame], 'image': images, 'label': label}

    def _load_data(self, ins, frame, t=1, global_n=0):
        r_im = os.path.join(DATA_ROOT, 'Processed_train/seq_{}/left_frames/frame{:03d}.png')
        r_lb = os.path.join(DATA_ROOT, 'Processed_train/seq_{}/labels/grayframe{:03d}.png') #512*640, class num

        if self.test:
            r_im = os.path.join(DATA_ROOT, 'Processed_test/seq_{}/left_frames/frame{:03d}.png') #resized image :512*640
            r_lb = os.path.join(DATA_ROOT, 'test/seq_{}/labels/frame{:03d}.png') #ori resolution

        imgs = []

        if t > frame: #when t > frame index, use future frame
            imgs += [Image.open(r_im.format(ins, i))
                     for i in range(frame+t-1, frame-1, -1)]
        else:
            imgs += [Image.open(r_im.format(ins, i))
                     for i in range(frame-t+1, frame+1)]

        for i in range(len(imgs)):
            imgs[i] = imgs[i].resize((self.crop_size['w'], self.crop_size['h']), Image.BILINEAR)


        if self.test:
            imgs = [np.array(i) for i in imgs]
            masks_color = np.asarray(Image.open(r_lb.format(ins, frame))) #1024,1280,4
            masks = np.zeros(masks_color.shape[:2])
            for i in range(self.class_num):
                masks[(masks_color[:,:,:3] == self.json_color[i]).sum(axis=-1) == 3] = i
            #print('!!',np.unique(masks))
            #print('Full resolution for test data!')
        elif self.mode == 'train':
            masks = Image.open(r_lb.format(ins, frame))
            masks = masks.resize((self.crop_size['w'], self.crop_size['h']), Image.NEAREST)
            #print('mask_size:',masks.size)
            #print('img_len:', len(imgs))
            #print('img_size:', imgs[0].size)
            # imgs = [np.array(i) for i in imgs]
            # masks = np.array(masks)
            # for i in range(len(imgs)):
            #     cv2.imwrite('../code_18_yu/inputvisualcheck/{}_image.png'.format(i), imgs[i])
            # cv2.imwrite('../code_18_yu/inputvisualcheck/mask.png', masks)
            # print(imgs[0].shape, masks.shape)
            # print(1 + '1')
            imgs, masks = self._random_scale(imgs, masks)

        return imgs, masks.astype('uint8')

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


# ----------------------------------------------------------------------------

# python -m dataset.Endovis2018 resize_dataset \
# -src="/raid/wjc/data/ead/ead2018/train" \
# -spt="/raid/wjc/data/ead/ead2018/Processed_train"

# python -m dataset.Endovis2018 resize_dataset \
# -src="/data3/zl/evs2018rss_raw/test" \
# -spt="/data3/zl/evs2018rss_resize/test"

def resize_dataset(src, spt):


    src = [src]
    dst = []
    while src:
        sub = src.pop()
        for item in os.listdir(sub):
            path = os.path.join(sub, item)
            if os.path.isdir(path):
                if item.startswith('seq_'):
                    dst.append(path)
                else:
                    src.append(path)

    for seq in tqdm(dst):
        for key in ['labels', 'left_frames']:
            raw_dir = os.path.join(seq, key)
            sav_dir = os.path.join(spt, os.path.basename(seq), key)
        
            file = [i for i in os.listdir(raw_dir) if i.startswith('frame')]
            print(key, len(file))
            assert len(file) == 149 or len(file) == 249 or len(file) == 250
            
            for item in file:
                raw_pt = os.path.join(raw_dir, item)
                sav_pt = os.path.join(sav_dir, item)
                
                img = cv2.imread(raw_pt)
                assert img.shape == (1024,1280,3)
                
                if key == 'labels':
                    img = img[::2,::2,:]
                else:
                    img = cv2.resize(
                        img, 
                        (1280//2,1024//2), 
                        interpolation=cv2.INTER_LINEAR,  # 双线性插值
                    )
                assert img.shape == (512,640,3)
                
                os.makedirs(os.path.dirname(sav_pt), exist_ok=True)
                cv2.imwrite(sav_pt, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])



def togray():
    mask_path = '../../2018data/ead2018/Processed_test/seq_1/labels/frame249.png'
    img = cv2.imread(mask_path)
    img2 = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
    mask_path2 = '../../2018data/ead2018/Processed_test/seq_1/labels/grayframe248.png'
    img_gray_sample = cv2.imread(mask_path2)
    sav_pt = '../../2018data/ead2018/Processed_test/seq_1/labels/grayframe249.png'
    cv2.imwrite(sav_pt, img_gray_sample, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    with open(LABEL_JSON, 'r') as f:
        lb_json = json.load(f)
    json_color = [item['color'] for item in lb_json]




if __name__ == '__main__':
    togray()
#    from fire import Fire
#     Fire()
#     loader = endovis2018('valid')
#     for d in loader:
# #         print(d['image'].shape,d['image'].max(),d['image'].min())
#         print(d['label'].shape)
#         break
