import io
import logging
import os
import time
import json
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import random
import torch
from contrast.cadis_visualization import *

lens = {"01": 177, "02": 170, "03": 176, "04": 173, "05": 178, "06": 183, "07": 183, "08": 184, "09": 175, "10": 205, "11": 182, "12": 217, "13": 171, \
        "14": 230, "15": 181, "16": 173, "17": 179, "18": 185, "19": 214, "20": 199, "21": 219, "22": 199, "23": 178, "24": 180, "25": 159}
train_videos = [1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25]
DATA_ROOT = 'xx/CaDis'
sub_type = {
    '1': remap_experiment1,
    '2': remap_experiment2,
    '3': remap_experiment3
}
sub_class_num = {'1':9, '2':18, '3':26}



def get_neg(ins):
    p = [1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25]
    p.remove(ins)
    neg1, neg2, neg3= random.sample(p, 3)
    frame1 = random.randint(0, lens[str(neg1).zfill(2)]-1)
    frame2 = random.randint(0, lens[str(neg2).zfill(2)]-1)
    frame3 = random.randint(0, lens[str(neg3).zfill(2)]-1)
    return neg1, neg2, neg3,  frame1, frame2, frame3

class pretrainDataset(data.Dataset):
    def __init__(self, transform=None, tag='1', size=480):
        super(pretrainDataset, self).__init__()
        self.size = size
        self.samples = [[v, f] for v in train_videos for f in range(0, lens[str(v).zfill(2)])]
        self.num_samples = len(self.samples)
        self.transform = transform
        self.remap_experiment = sub_type[tag]
        self.class_num = sub_class_num[tag]
        # self.mask_transform = mask_transform
        print('DATA: Cata')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ins, frame = self.samples[idx]
        image, image_v,image_1, image_2, image_3, image_4, image_neg1,image_neg1p1,image_neg2p2,image_neg2p3, image_neg2, image_neg2p1,image_neg2p2,image_neg2p3, image_neg3, image_neg3p1,image_neg3p2,image_neg3p3, label,label_v,label_1,label_neg1,label_neg2,label_neg3 \
            = self._load_data(ins,frame)
        if self.transform is not None:
            img,img1,img2,img3, labelori, coord = self.transform[0](image,image_1,image_2,image_3, label) #tensor vector ??
            seq = self.append_img_1(img3,img2,img1,img)
            img1,img2,img3,img_v, labelv, coord_v = self.transform[1](image_1,image_2,image_3,image_v, label_v) #tensor vector
            seq_v = self.append_img_1(img3,img2,img1,img_v)
            img1,img2,img3,img4, label1, coord1 = self.transform[2](image_1,image_2,image_3,image_4, label_1) #tensor vector
            seq_1 = self.append_img_1(img4,img3,img2,img1)

            img_neg,img_negp1,img_negp2,img_negp3, labelneg1, coordneg1 = self.transform[3](image_neg1,image_neg1p1,image_neg2p2,image_neg2p3,label_neg1) #tensor vector
            seq_2 = self.append_img_1(img_negp3,img_negp2,img_negp1,img_neg)
            img_neg2,img_neg2p1,img_neg2p2,img_neg2p3, labelneg2, coordneg2 = self.transform[4](image_neg2, image_neg2p1,image_neg2p2,image_neg2p3,label_neg2) #tensor vector
            seq_3 = self.append_img_1(img_neg2p3,img_neg2p2,img_neg2p1,img_neg2)
            img_neg3,img_neg3p1,img_neg3p2,img_neg3p3, labelneg3, coordneg3 = self.transform[5](image_neg3, image_neg3p1, image_neg3p2, image_neg3p3,label_neg3) #tensor vector
            seq_4 = self.append_img_1(img_neg3p3,img_neg3p2,img_neg3p1,img_neg3)

            return seq, seq_v, seq_1, seq_2, seq_3, seq_4, labelori, labelv, label1, labelneg1, labelneg2, labelneg3

        else:
            img = image
            return img

    def _load_data(self, ins, frame):

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

        image = self.load_resize_img(ins,frame)
        label = self.load_resize_lb(ins,frame)

        image_1 = self.load_resize_img(ins,prev1)
        label_1 = self.load_resize_lb(ins,prev1)

        image_2 = self.load_resize_img(ins,prev2)
        image_3 = self.load_resize_img(ins,prev3)
        image_4 = self.load_resize_img(ins,prev4)

        image_neg1 = self.load_resize_img(neg1,neg_frame1)
        label_neg1 = self.load_resize_lb(neg1,neg_frame1)
        image_neg1p1 = self.load_resize_img(neg1,neg1_prev1)
        image_neg1p2 = self.load_resize_img(neg1,neg1_prev2)
        image_neg1p3 = self.load_resize_img(neg1,neg1_prev3)

        image_neg2 = self.load_resize_img(neg2,neg_frame2)
        label_neg2 = self.load_resize_lb(neg2,neg_frame2)
        image_neg2p1 = self.load_resize_img(neg2,neg2_prev1)
        image_neg2p2 = self.load_resize_img(neg2,neg2_prev2)
        image_neg2p3 = self.load_resize_img(neg2,neg2_prev3)

        image_neg3 = self.load_resize_img(neg3,neg_frame3)
        label_neg3 = self.load_resize_lb(neg3,neg_frame3)
        image_neg3p1 = self.load_resize_img(neg3,neg3_prev1)
        image_neg3p2 = self.load_resize_img(neg3,neg3_prev2)
        image_neg3p3 = self.load_resize_img(neg3,neg3_prev3)

        return image, image, image_1, image_2, image_3,image_4, image_neg1,image_neg1p1,image_neg1p2,image_neg1p3, image_neg2, image_neg2p1,image_neg2p2,image_neg2p3, image_neg3, image_neg3p1,image_neg3p2,image_neg3p3, label,label,label_1,label_neg1,label_neg2,label_neg3

    def append_img_1(self,im1,im2,im3,im4):

        seq = []
        seq.append(im1.unsqueeze(0)) #3,224,224
        seq.append(im2.unsqueeze(0))
        seq.append(im3.unsqueeze(0))
        seq.append(im4.unsqueeze(0)) #a list
        seq = torch.cat(seq,dim=0)
        return seq

    def load_resize_img(self,ins,frame):
        r_im = os.path.join(DATA_ROOT, 'Video{}/Images/Video{}_frame{}.png')
        im = Image.open(r_im.format(str(ins).zfill(2), str(ins).zfill(2), str(frame).zfill(6)))
        im = im.resize((480, 270), Image.BILINEAR)
        return im

    def load_resize_lb(self,ins,frame):
        r_lb = os.path.join(DATA_ROOT, 'Video{}/Labels/Video{}_frame{}.png')
        lb = Image.open(r_lb.format(str(ins).zfill(2), str(ins).zfill(2), str(frame).zfill(6)))
        lb = np.asarray(lb.resize((480, 270), Image.NEAREST))
        lb2, _, _ = self.remap_experiment(lb)
        lb2[lb2==255] = self.class_num-1
        lb2 = Image.fromarray(lb2)


        return lb2
