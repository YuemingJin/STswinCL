import io
import logging
import os
import time
import json
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import torch
import cv2
from torchvision import transforms
import torchvision.utils as tvutil

DATA_ROOT = 'xxx/scene'
Procedures = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]



def get_neg(ins):
    p = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
    p.remove(ins)
    neg1, neg2, neg3 = random.sample(p, 3)
    frame1 = random.randint(0, 148)
    frame2 = random.randint(0, 148)
    frame3 = random.randint(0, 148)
    return neg1, neg2, neg3, frame1, frame2, frame3

class pretrainDataset(data.Dataset):
    def __init__(self, transform=None, size=480):
        super(pretrainDataset, self).__init__()
        self.size = size
        self.samples = [[f,i] for i in range(0, 149) for f in Procedures]
        self.num_samples = len(self.samples)
        self.transform = transform
        # self.mask_transform = mask_transform
        print('DATA: Endovis2018')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ins, frame = self.samples[idx]


        image, image_v,image_1, image_2, image_3, image_4, image_neg1,image_neg1p1,image_neg1p2,image_neg1p3, image_neg2, image_neg2p1,image_neg2p2,image_neg2p3, image_neg3, image_neg3p1,image_neg3p2,image_neg3p3, label,label_v,label_1,label_neg1,label_neg2,label_neg3 \
            = self._load_data(ins,frame)

        # current frame, adjacent frame, four negative frames from other videos
        t = 4
        c = 3
        h=w=480

        if self.transform is not None:
            img,img1,img2,img3, labelori, coord = self.transform[0](image,image_1,image_2,image_3, label) #tensor vector ??
            seq = self.append_img_1(img3,img2,img1,img) # tensor with size (t,c,h,w)
            img1,img2,img3,img_v, labelv, coord_v = self.transform[1](image_1,image_2,image_3,image_v, label_v) #tensor vector
            seq_v = self.append_img_1(img3,img2,img1,img_v)
            img1,img2,img3,img4, label1, coord1 = self.transform[2](image_1,image_2,image_3,image_4, label_1) #tensor vector
            seq_1 = self.append_img_1(img4,img3,img2,img1)

            img_neg,img_negp1,img_negp2,img_negp3, labelneg1, coordneg1 = self.transform[3](image_neg1,image_neg1p1,image_neg1p2,image_neg1p3,label_neg1) #tensor vector
            seq_2 = self.append_img_1(img_negp3,img_negp2,img_negp1,img_neg)
            img_neg2,img_neg2p1,img_neg2p2,img_neg2p3, labelneg2, coordneg2 = self.transform[4](image_neg2, image_neg2p1,image_neg2p2,image_neg2p3,label_neg2) #tensor vector
            seq_3 = self.append_img_1(img_neg2p3,img_neg2p2,img_neg2p1,img_neg2)
            img_neg3,img_neg3p1,img_neg3p2,img_neg3p3, labelneg3, coordneg3 = self.transform[5](image_neg3, image_neg3p1, image_neg3p2, image_neg3p3,label_neg3) #tensor vector
            seq_4 = self.append_img_1(img_neg3p3,img_neg3p2,img_neg3p1,img_neg3)

            return seq,seq_v, seq_1, seq_2,seq_3,seq_4, labelori, labelv, label1, labelneg1,labelneg2,labelneg3

        else:
            img = image
            return img

    def _load_data(self, ins, frame):
        r_im = os.path.join(DATA_ROOT, 'Processed_train/seq_{}/left_frames/frame{:03d}.png')
        r_lb = os.path.join(DATA_ROOT, 'Processed_train/seq_{}/labels/grayframe{:03d}.png')

        #t=4, need 6 frames
        t = 2
        tt=4
        if tt > frame: #when t > frame index, use future frame
            ind = []
            ind += [i
                    for i in range(frame+tt-1, frame-1, -1)]
            prev1 = ind[0]
            prev2 = ind[1]
            prev3 = ind[2]
            prev4 = ind[3]
            frame = prev1+1


        else:
            prev1 = frame-1 #larger
            prev2 = frame-2
            prev3 = frame-3
            prev4 = frame-4

        ttt = 3
        neg1, neg2, neg3, neg_frame1, neg_frame2, neg_frame3 = get_neg(ins)
        if ttt > neg_frame1: #when t > frame index, use future frame
            ind = []
            ind += [i
                    for i in range(neg_frame1+ttt-1, neg_frame1-1, -1)]
            neg1_prev1 = ind[0]
            neg1_prev2 = ind[1]
            neg1_prev3 = ind[2]
            neg_frame1 = neg1_prev1+1
        else:
            neg1_prev1 = neg_frame1-1 #larger
            neg1_prev2 = neg_frame1-2
            neg1_prev3 = neg_frame1-3

        if ttt > neg_frame2: #when t > frame index, use future frame
            ind = []
            ind += [i
                    for i in range(neg_frame2+ttt-1, neg_frame2-1, -1)]
            neg2_prev1 = ind[0]
            neg2_prev2 = ind[1]
            neg2_prev3 = ind[2]
            neg_frame2 = neg2_prev1+1
        else:
            neg2_prev1 = neg_frame2-1 #larger
            neg2_prev2 = neg_frame2-2
            neg2_prev3 = neg_frame2-3

        if ttt > neg_frame3: #when t > frame index, use future frame
            ind = []
            ind += [i
                    for i in range(neg_frame3+ttt-1, neg_frame3-1, -1)]
            neg3_prev1 = ind[0]
            neg3_prev2 = ind[1]
            neg3_prev3 = ind[2]
            neg_frame3 = neg3_prev1+1
        else:
            neg3_prev1 = neg_frame3-1 #larger
            neg3_prev2 = neg_frame3-2
            neg3_prev3 = neg_frame3-3


        image = Image.open(r_im.format(ins, frame))
        image = image.resize((480, 270), Image.BILINEAR)
        label = Image.open(r_lb.format(ins, frame))
        label = label.resize((480, 270), Image.NEAREST)

        image_1 = Image.open(r_im.format(ins, prev1))
        image_1 = image_1.resize((480, 270), Image.BILINEAR)
        label_1 = Image.open(r_lb.format(ins, prev1))
        label_1 = label_1.resize((480, 270), Image.NEAREST)

        image_2 = Image.open(r_im.format(ins, prev2))
        image_2 = image_2.resize((480, 270), Image.BILINEAR)

        image_3 = Image.open(r_im.format(ins, prev3))
        image_3 = image_3.resize((480, 270), Image.BILINEAR)

        image_4 = Image.open(r_im.format(ins, prev4))
        image_4 = image_4.resize((480, 270), Image.BILINEAR)

        image_neg1 = Image.open(r_im.format(neg1, neg_frame1))
        image_neg1 = image_neg1.resize((480, 270), Image.BILINEAR)
        label_neg1 = Image.open(r_lb.format(neg1, neg_frame1))
        label_neg1 = label_neg1.resize((480, 270), Image.NEAREST)
        image_neg1p1 = Image.open(r_im.format(neg1, neg1_prev1))
        image_neg1p1 = image_neg1p1.resize((480, 270), Image.BILINEAR)
        image_neg1p2 = Image.open(r_im.format(neg1, neg1_prev2))
        image_neg1p2 = image_neg1p2.resize((480, 270), Image.BILINEAR)
        image_neg1p3 = Image.open(r_im.format(neg1, neg1_prev3))
        image_neg1p3 = image_neg1p3.resize((480, 270), Image.BILINEAR)

        image_neg2 = Image.open(r_im.format(neg2, neg_frame2))
        image_neg2 = image_neg2.resize((480, 270), Image.BILINEAR)
        label_neg2 = Image.open(r_lb.format(neg2, neg_frame2))
        label_neg2 = label_neg2.resize((480, 270), Image.NEAREST)
        image_neg2p1 = Image.open(r_im.format(neg2, neg2_prev1))
        image_neg2p1 = image_neg2p1.resize((480, 270), Image.BILINEAR)
        image_neg2p2 = Image.open(r_im.format(neg2, neg2_prev2))
        image_neg2p2 = image_neg2p2.resize((480, 270), Image.BILINEAR)
        image_neg2p3 = Image.open(r_im.format(neg2, neg2_prev3))
        image_neg2p3 = image_neg2p3.resize((480, 270), Image.BILINEAR)

        image_neg3 = Image.open(r_im.format(neg3, neg_frame3))
        image_neg3 = image_neg3.resize((480, 270), Image.BILINEAR)
        label_neg3 = Image.open(r_lb.format(neg3, neg_frame3))
        label_neg3 = label_neg3.resize((480, 270), Image.NEAREST)
        image_neg3p1 = Image.open(r_im.format(neg3, neg3_prev1))
        image_neg3p1 = image_neg3p1.resize((480, 270), Image.BILINEAR)
        image_neg3p2 = Image.open(r_im.format(neg3, neg3_prev2))
        image_neg3p2 = image_neg3p2.resize((480, 270), Image.BILINEAR)
        image_neg3p3 = Image.open(r_im.format(neg3, neg3_prev3))
        image_neg3p3 = image_neg3p3.resize((480, 270), Image.BILINEAR)

        return image, image, image_1, image_2, image_3,image_4, image_neg1,image_neg1p1,image_neg1p2,image_neg1p3, image_neg2, image_neg2p1,image_neg2p2,image_neg2p3, image_neg3, image_neg3p1,image_neg3p2,image_neg3p3, label,label,label_1,label_neg1,label_neg2,label_neg3


    def append_img_1(self,im1,im2,im3,im4):

        seq = []
        seq.append(im1.unsqueeze(0)) #3,224,224
        seq.append(im2.unsqueeze(0))
        seq.append(im3.unsqueeze(0))
        seq.append(im4.unsqueeze(0)) #a list
        seq = torch.cat(seq,dim=0)
        return seq

