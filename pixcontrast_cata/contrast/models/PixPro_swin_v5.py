import numpy as np
import sys
import os
dirname = os.path.dirname(__file__)
sys.path.append(dirname)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size

from Ours.base import TswinPlusv5
from Ours.LoadModel import load_model_full
from contrast.option import parse_option

num_class_table = {'1':9, '2':18, '3':26} #add the ignore class

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x

def posMask(pred1, pred2,class_num):

    B, C, H, W = pred1.shape #C=1
    pred1 = pred1.reshape(B, H*W) #B,HW
    pred2 = pred2.reshape(B, H*W) #B,HW
    # print(torch.max(pred2))
    pred1_onehot = F.one_hot(pred1.long(), num_classes=class_num).float().cuda() #B,HW,12
    pred2_onehot = F.one_hot(pred2.long(), num_classes=class_num).float().cuda()
    mask = torch.bmm(pred1_onehot, pred2_onehot.transpose(1, 2)) #B,HWHW
    return mask

def negMask(pred1, pred2,class_num):

    B, C, H, W = pred1.shape
    pred1 = pred1.reshape(B, H*W)
    pred2 = pred2.reshape(B, H*W)
    # print(torch.max(pred2))
    pred1_onehot = F.one_hot(pred1.long(), num_classes=class_num).float().cuda()
    pred2_onehot = F.one_hot(pred2.long(), num_classes=class_num).float().cuda()
    mask = torch.bmm(pred1_onehot, pred2_onehot.transpose(1, 2))
    return 1 - mask


def regression_loss(q, k, adj1, adj2, adj3,neg3, label_patch1, label_patch2, label_adj1, label_adj2, label_adj3,label_neg3,class_num):
    N, C, H, W = q.shape
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)
    adj1 = adj1.view(N, C, -1)
    adj2 = adj2.view(N, C, -1)
    adj3 = adj3.view(N, C, -1)
    neg3 = neg3.view(N, C, -1)

    #matrix multiplication
    logit1 = torch.bmm(q.transpose(1, 2), k)
    logit2 = torch.bmm(q.transpose(1, 2), adj1)
    logit3 = torch.bmm(q.transpose(1, 2), adj2)
    logit4 = torch.bmm(q.transpose(1, 2), adj3)
    logit5 = torch.bmm(q.transpose(1, 2), neg3)

    mask_p = posMask(label_patch1, label_patch2,class_num).cuda().detach()
    mask_p2 = posMask(label_patch1, label_adj1,class_num).cuda().detach()
    mask_p3 = posMask(label_patch1, label_adj2,class_num).cuda().detach()
    mask_p4 = posMask(label_patch1, label_adj3,class_num).cuda().detach()
    mask_p5 = posMask(label_patch1, label_neg3,class_num).cuda().detach()


    # print(mask_p.mean())
    mask_n1 = mask_p.clone()
    mask_n1[mask_p==0] = 1
    mask_n1[mask_p!=0] = 0
    mask_n2 = negMask(label_patch1, label_adj1,class_num).cuda().detach()
    mask_n3 = negMask(label_patch1, label_adj2,class_num).cuda().detach()
    mask_n4 = negMask(label_patch1, label_adj3,class_num).cuda().detach()
    mask_n5 = negMask(label_patch1, label_neg3,class_num).cuda().detach()

    masked_p = mask_p * logit1 #B,HW,HW
    masked_p2 = mask_p2 * logit2
    masked_p3 = mask_p3 * logit3
    masked_p4 = mask_p4 * logit4
    masked_p5 = mask_p5 * logit5

    masked_n1 = mask_n1 * logit1
    masked_n2 = mask_n2 * logit2
    masked_n3 = mask_n3 * logit3
    masked_n4 = mask_n4 * logit4
    masked_n5 = mask_n5 * logit5

    P = torch.sum(masked_p+masked_p2+masked_p3+masked_p4+masked_p5, dim=-1) / (torch.sum(mask_p+mask_p2+mask_p3+mask_p4+mask_p5, dim=-1) + 1e-6)

    N = torch.sum(masked_n1, dim=-1) / (torch.sum(mask_n1, dim=-1) + 1e-6) + torch.sum(masked_n2, dim=-1) / (torch.sum(mask_n2, dim=-1) + 1e-6) \
        + torch.sum(masked_n3, dim=-1) / (torch.sum(mask_n3, dim=-1) + 1e-6) + torch.sum(masked_n4, dim=-1) / (torch.sum(mask_n4, dim=-1) + 1e-6) \
        + torch.sum(masked_n5, dim=-1) / (torch.sum(mask_n5, dim=-1) + 1e-6)
    P_exp = torch.exp(P)
    N_exp = torch.exp(N)
    tem = torch.log(P_exp / (P_exp + N_exp)+1e-6)
    loss = -torch.mean(torch.log(P_exp / (P_exp + N_exp)+1e-6))
    return loss

def Proj_Head(in_dim=400, inner_dim=512, out_dim=256): ##!!!
    return MLP2d(in_dim, inner_dim, out_dim)

def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)




class PixPro(nn.Module):
    def __init__(self, args):
        super(PixPro, self).__init__()
        print('!!!Pixpro_swin version')

        # parse arguments
        self.pixpro_p               = args.pixpro_p
        self.pixpro_momentum        = args.pixpro_momentum
        # self.pixpro_pos_ratio       = args.pixpro_pos_ratio
        self.pixpro_clamp_value     = args.pixpro_clamp_value
        self.pixpro_transform_layer = args.pixpro_transform_layer
        self.pixpro_ins_loss_weight = args.pixpro_ins_loss_weight

        # create the encoder
        if args.data == 'endo18':
            model_path = 'xx/results/'+args.pretrainpth #!!!
            class_num = 12
        elif args.data == 'cata':
            model_path = 'xx/results/'+args.pretrainpth  # !!!
            class_num = int(num_class_table[args.tag])
        print('Pretrain model from:',model_path)
        print('Class_num:', class_num)
        self.segmentor = TswinPlusv5(class_num).cuda()
        self.segmentor = load_model_full(self.segmentor, model_path)


        self.encoder_1 = self.segmentor.resnet ###
        self.encoder_2 = self.segmentor.swin
        self.encoder_3 = self.segmentor.aspp
        self.proj1 = self.segmentor.project1
        self.proj2 = self.segmentor.project2
        self.proj3 = self.segmentor.project3
        del self.segmentor
        self.projector = Proj_Head()

        # create the encoder_k
        self.segmentor1 = TswinPlusv5(class_num).cuda()
        self.segmentor1 = load_model_full(self.segmentor1, model_path)
        self.encoder_k_1 = self.segmentor1.resnet
        self.encoder_k_2 = self.segmentor1.swin
        self.encoder_k_3 = self.segmentor1.aspp
        self.proj_k_1 = self.segmentor1.project1
        self.proj_k_2 = self.segmentor1.project2
        self.proj_k_3 = self.segmentor1.project3
        del self.segmentor1
        # self.encoder_k = base_encoder(head_type='early_return')
        self.projector_k = Proj_Head()

        for param_q, param_k in zip(self.encoder_1.parameters(), self.encoder_k_1.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.encoder_2.parameters(), self.encoder_k_2.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.encoder_3.parameters(), self.encoder_k_3.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.proj1.parameters(), self.proj_k_1.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.proj2.parameters(), self.proj_k_2.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.proj3.parameters(), self.proj_k_3.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_1)  ##for batch normalization in different GPUs, calculate the mean and std in all GPUs
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_2)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_3)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.proj1)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.proj2)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.proj3)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k_1)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k_2)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k_3)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.proj_k_1)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.proj_k_2)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.proj_k_3)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)

        self.K = int(args.num_instances * 1. / get_world_size() / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / get_world_size() / args.batch_size * (args.start_epoch - 1))
        # self.K = 0
        # self.k = 1
        if self.pixpro_transform_layer == 0:
            self.value_transform = Identity()
        elif self.pixpro_transform_layer == 1:
            self.value_transform = conv1x1(in_planes=256, out_planes=256)
        elif self.pixpro_transform_layer == 2:
            self.value_transform = MLP2d(in_dim=256, inner_dim=256, out_dim=256)
        else:
            raise NotImplementedError

        if self.pixpro_ins_loss_weight > 0.:
            self.projector_instance = Proj_Head()
            self.projector_instance_k = Proj_Head()
            self.predictor = Pred_Head()

            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance_k)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

            self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.pixpro_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        for param_q, param_k in zip(self.encoder_1.parameters(), self.encoder_k_1.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.encoder_2.parameters(), self.encoder_k_2.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.encoder_3.parameters(), self.encoder_k_3.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.proj1.parameters(), self.proj_k_1.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.proj2.parameters(), self.proj_k_2.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.proj3.parameters(), self.proj_k_3.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        if self.pixpro_ins_loss_weight > 0.:
            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    def forward(self, seq_1, seq_2, seq_3, seq_4, seq_5, seq_6):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        #print('newone!!!!!!!')
        t = 4

        encoded_f = []
        for i in range(t):
            tensor = self.encoder_1(seq_1[:, i])
            encoded_f.append(tensor.unsqueeze(1))  # 2,1,512,64,80
        encoded_seq = torch.cat(encoded_f, dim=1)
        res_output = encoded_seq[:, -1, :, :, :]

        tem_seq_1, tem_seq_2 = self.encoder_2(encoded_seq)  # swin
        tem_output_1 = tem_seq_1[:, -1, :, :, :]
        tem_output_2 = tem_seq_2[:, -1, :, :, :]

        aspp_output = self.encoder_3(tem_output_2)  # aspp
        # print('aspp shape:', encoded_seq_1.shape)

        res_output_pro = self.proj1(res_output)
        tem_output_1_pro = self.proj2(tem_output_1)
        tem_output_2_pro = self.proj3(tem_output_2)

        tem_output_2_pro = F.upsample(tem_output_2_pro, size=res_output_pro.shape[2:], mode='bilinear',
                                      align_corners=False)
        aspp_output = F.upsample(aspp_output, size=res_output_pro.shape[2:], mode='bilinear',
                                 align_corners=False)
        output_seq = torch.cat([res_output_pro, tem_output_1_pro, tem_output_2_pro, aspp_output], dim=1)  # 2,12,64,80
        proj = self.projector(output_seq)
        # print('projector shape:', proj_1.shape)
        pred_1 = F.normalize(proj, dim=1)

        encoded_f = []
        for i in range(t):
            tensor = self.encoder_1(seq_2[:, i])
            # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
            encoded_f.append(tensor.unsqueeze(1))  # 2,1,512,64,80
        encoded_seq = torch.cat(encoded_f, dim=1)
        # print('backbone shape:', encoded_seq_1.shape)#b,t,c,h,w: 4,4,512,28,28
        res_output = encoded_seq[:, -1, :, :, :]

        tem_seq_1, tem_seq_2 = self.encoder_2(encoded_seq)  # swin
        tem_output_1 = tem_seq_1[:, -1, :, :, :]
        tem_output_2 = tem_seq_2[:, -1, :, :, :]

        aspp_output = self.encoder_3(tem_output_2)  # aspp
        # print('aspp shape:', encoded_seq_1.shape)

        res_output_pro = self.proj1(res_output)  # 2,48,64,80
        tem_output_1_pro = self.proj2(tem_output_1)
        tem_output_2_pro = self.proj3(tem_output_2)

        tem_output_2_pro = F.upsample(tem_output_2_pro, size=res_output_pro.shape[2:], mode='bilinear',
                                      align_corners=False)
        aspp_output = F.upsample(aspp_output, size=res_output_pro.shape[2:], mode='bilinear',
                                 align_corners=False)  # 2,256,64,80
        output_seq = torch.cat([res_output_pro, tem_output_1_pro, tem_output_2_pro, aspp_output], dim=1)  # 2,12,64,80
        # print('concat shape:', output_seq_1.shape) #4,304,28,28
        proj = self.projector(output_seq)
        # print('projector shape:', proj_1.shape)
        pred_2 = F.normalize(proj, dim=1)
        # print('final shape:', pred_1.shape) #2,256,28,28

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            t = 4
            # print('seq shape:',seq_1.shape)#b,t,c,w,h:2,4,3,224,224
            encoded_f_ng = []
            for i in range(t):
                tensor = self.encoder_k_1(seq_1[:, i])
                # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
                encoded_f_ng.append(tensor.unsqueeze(1))  # 2,1,512,64,80
            encoded_seq_ng = torch.cat(encoded_f_ng, dim=1)
            # print('backbone shape:', encoded_seq_1.shape)#b,t,c,h,w: 4,4,512,28,28
            res_output_ng = encoded_seq_ng[:, -1, :, :, :]

            tem_seq_1_ng, tem_seq_2_ng = self.encoder_k_2(encoded_seq_ng)  # swin
            tem_output_1_ng = tem_seq_1_ng[:, -1, :, :, :]
            tem_output_2_ng = tem_seq_2_ng[:, -1, :, :, :]

            aspp_output_ng = self.encoder_k_3(tem_output_2_ng)  # aspp
            # print('aspp shape:', encoded_seq_1.shape)

            res_output_pro_ng = self.proj_k_1(res_output_ng)  # 2,48,64,80
            tem_output_1_pro_ng = self.proj_k_2(tem_output_1_ng)
            tem_output_2_pro_ng = self.proj_k_3(tem_output_2_ng)

            tem_output_2_pro_ng = F.upsample(tem_output_2_pro_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                             align_corners=False)
            aspp_output_ng = F.upsample(aspp_output_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                        align_corners=False)  # 2,256,64,80
            output_seq_ng = torch.cat([res_output_pro_ng, tem_output_1_pro_ng, tem_output_2_pro_ng, aspp_output_ng],
                                      dim=1)  # 2,12,64,80
            # print('concat shape:', output_seq_1.shape) #4,304,28,28
            proj_ng = self.projector_k(output_seq_ng)
            # print('projector shape:', proj_1.shape)
            proj_1_ng = F.normalize(proj_ng, dim=1)

            ###---------------
            encoded_f_ng = []
            for i in range(t):
                tensor = self.encoder_k_1(seq_2[:, i])
                # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
                encoded_f_ng.append(tensor.unsqueeze(1))  # 2,1,512,64,80
            encoded_seq_ng = torch.cat(encoded_f_ng, dim=1)
            # print('backbone shape:', encoded_seq_1.shape)#b,t,c,h,w: 4,4,512,28,28
            res_output_ng = encoded_seq_ng[:, -1, :, :, :]

            tem_seq_1_ng, tem_seq_2_ng = self.encoder_k_2(encoded_seq_ng)  # swin
            tem_output_1_ng = tem_seq_1_ng[:, -1, :, :, :]
            tem_output_2_ng = tem_seq_2_ng[:, -1, :, :, :]

            aspp_output_ng = self.encoder_k_3(tem_output_2_ng)  # aspp
            # print('aspp shape:', encoded_seq_1.shape)

            res_output_pro_ng = self.proj_k_1(res_output_ng)  # 2,48,64,80
            tem_output_1_pro_ng = self.proj_k_2(tem_output_1_ng)
            tem_output_2_pro_ng = self.proj_k_3(tem_output_2_ng)

            tem_output_2_pro_ng = F.upsample(tem_output_2_pro_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                             align_corners=False)
            aspp_output_ng = F.upsample(aspp_output_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                        align_corners=False)  # 2,256,64,80
            output_seq_ng = torch.cat([res_output_pro_ng, tem_output_1_pro_ng, tem_output_2_pro_ng, aspp_output_ng],
                                      dim=1)  # 2,12,64,80
            # print('concat shape:', output_seq_1.shape) #4,304,28,28
            proj_ng = self.projector_k(output_seq_ng)
            # print('projector shape:', proj_1.shape)
            proj_2_ng = F.normalize(proj_ng, dim=1)

            encoded_f_ng = []
            for i in range(t):
                tensor = self.encoder_k_1(seq_3[:, i])
                # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
                encoded_f_ng.append(tensor.unsqueeze(1))  # 2,1,512,64,80
            encoded_seq_ng = torch.cat(encoded_f_ng, dim=1)
            # print('backbone shape:', encoded_seq_1.shape)#b,t,c,h,w: 4,4,512,28,28
            res_output_ng = encoded_seq_ng[:, -1, :, :, :]

            tem_seq_1_ng, tem_seq_2_ng = self.encoder_k_2(encoded_seq_ng)  # swin
            tem_output_1_ng = tem_seq_1_ng[:, -1, :, :, :]
            tem_output_2_ng = tem_seq_2_ng[:, -1, :, :, :]

            aspp_output_ng = self.encoder_k_3(tem_output_2_ng)  # aspp
            # print('aspp shape:', encoded_seq_1.shape)

            res_output_pro_ng = self.proj_k_1(res_output_ng)  # 2,48,64,80
            tem_output_1_pro_ng = self.proj_k_2(tem_output_1_ng)
            tem_output_2_pro_ng = self.proj_k_3(tem_output_2_ng)

            tem_output_2_pro_ng = F.upsample(tem_output_2_pro_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                             align_corners=False)
            aspp_output_ng = F.upsample(aspp_output_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                        align_corners=False)  # 2,256,64,80
            output_seq_ng = torch.cat([res_output_pro_ng, tem_output_1_pro_ng, tem_output_2_pro_ng, aspp_output_ng],
                                      dim=1)  # 2,12,64,80
            # print('concat shape:', output_seq_1.shape) #4,304,28,28
            proj_ng = self.projector_k(output_seq_ng)
            # print('projector shape:', proj_1.shape)
            proj_adj1_ng = F.normalize(proj_ng, dim=1)

            encoded_f_ng = []
            for i in range(t):
                tensor = self.encoder_k_1(seq_4[:, i])
                # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
                encoded_f_ng.append(tensor.unsqueeze(1))  # 2,1,512,64,80
            encoded_seq_ng = torch.cat(encoded_f_ng, dim=1)
            # print('backbone shape:', encoded_seq_1.shape)#b,t,c,h,w: 4,4,512,28,28
            res_output_ng = encoded_seq_ng[:, -1, :, :, :]

            tem_seq_1_ng, tem_seq_2_ng = self.encoder_k_2(encoded_seq_ng)  # swin
            tem_output_1_ng = tem_seq_1_ng[:, -1, :, :, :]
            tem_output_2_ng = tem_seq_2_ng[:, -1, :, :, :]

            aspp_output_ng = self.encoder_k_3(tem_output_2_ng)  # aspp
            # print('aspp shape:', encoded_seq_1.shape)

            res_output_pro_ng = self.proj_k_1(res_output_ng)  # 2,48,64,80
            tem_output_1_pro_ng = self.proj_k_2(tem_output_1_ng)
            tem_output_2_pro_ng = self.proj_k_3(tem_output_2_ng)

            tem_output_2_pro_ng = F.upsample(tem_output_2_pro_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                             align_corners=False)
            aspp_output_ng = F.upsample(aspp_output_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                        align_corners=False)  # 2,256,64,80
            output_seq_ng = torch.cat([res_output_pro_ng, tem_output_1_pro_ng, tem_output_2_pro_ng, aspp_output_ng],
                                      dim=1)  # 2,12,64,80
            # print('concat shape:', output_seq_1.shape) #4,304,28,28
            proj_ng = self.projector_k(output_seq_ng)
            # print('projector shape:', proj_1.shape)
            proj_adj2_ng = F.normalize(proj_ng, dim=1)


            encoded_f_ng = []
            for i in range(t):
                tensor = self.encoder_k_1(seq_5[:, i])
                # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
                encoded_f_ng.append(tensor.unsqueeze(1))  # 2,1,512,64,80
            encoded_seq_ng = torch.cat(encoded_f_ng, dim=1)
            # print('backbone shape:', encoded_seq_1.shape)#b,t,c,h,w: 4,4,512,28,28
            res_output_ng = encoded_seq_ng[:, -1, :, :, :]

            tem_seq_1_ng, tem_seq_2_ng = self.encoder_k_2(encoded_seq_ng)  # swin
            tem_output_1_ng = tem_seq_1_ng[:, -1, :, :, :]
            tem_output_2_ng = tem_seq_2_ng[:, -1, :, :, :]

            aspp_output_ng = self.encoder_k_3(tem_output_2_ng)  # aspp
            # print('aspp shape:', encoded_seq_1.shape)

            res_output_pro_ng = self.proj_k_1(res_output_ng)  # 2,48,64,80
            tem_output_1_pro_ng = self.proj_k_2(tem_output_1_ng)
            tem_output_2_pro_ng = self.proj_k_3(tem_output_2_ng)

            tem_output_2_pro_ng = F.upsample(tem_output_2_pro_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                             align_corners=False)
            aspp_output_ng = F.upsample(aspp_output_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                        align_corners=False)  # 2,256,64,80
            output_seq_ng = torch.cat([res_output_pro_ng, tem_output_1_pro_ng, tem_output_2_pro_ng, aspp_output_ng],
                                      dim=1)  # 2,12,64,80
            # print('concat shape:', output_seq_1.shape) #4,304,28,28
            proj_ng = self.projector_k(output_seq_ng)
            # print('projector shape:', proj_1.shape)
            proj_adj3_ng = F.normalize(proj_ng, dim=1)

            encoded_f_ng = []
            for i in range(t):
                tensor = self.encoder_k_1(seq_6[:, i])
                # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
                encoded_f_ng.append(tensor.unsqueeze(1))  # 2,1,512,64,80
            encoded_seq_ng = torch.cat(encoded_f_ng, dim=1)
            # print('backbone shape:', encoded_seq_1.shape)#b,t,c,h,w: 4,4,512,28,28
            res_output_ng = encoded_seq_ng[:, -1, :, :, :]

            tem_seq_1_ng, tem_seq_2_ng = self.encoder_k_2(encoded_seq_ng)  # swin
            tem_output_1_ng = tem_seq_1_ng[:, -1, :, :, :]
            tem_output_2_ng = tem_seq_2_ng[:, -1, :, :, :]

            aspp_output_ng = self.encoder_k_3(tem_output_2_ng)  # aspp
            # print('aspp shape:', encoded_seq_1.shape)

            res_output_pro_ng = self.proj_k_1(res_output_ng)  # 2,48,64,80
            tem_output_1_pro_ng = self.proj_k_2(tem_output_1_ng)
            tem_output_2_pro_ng = self.proj_k_3(tem_output_2_ng)

            tem_output_2_pro_ng = F.upsample(tem_output_2_pro_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                             align_corners=False)
            aspp_output_ng = F.upsample(aspp_output_ng, size=res_output_pro_ng.shape[2:], mode='bilinear',
                                        align_corners=False)  # 2,256,64,80
            output_seq_ng = torch.cat([res_output_pro_ng, tem_output_1_pro_ng, tem_output_2_pro_ng, aspp_output_ng],
                                      dim=1)  # 2,12,64,80
            # print('concat shape:', output_seq_1.shape) #4,304,28,28
            proj_ng = self.projector_k(output_seq_ng)
            # print('projector shape:', proj_1.shape)
            proj_neg3_ng = F.normalize(proj_ng, dim=1)



        return pred_1, pred_2, proj_1_ng, proj_2_ng, proj_adj1_ng, proj_adj2_ng, proj_adj3_ng, proj_neg3_ng



class ConsistencyLoss(nn.Module):
    def __init__(self, args):
        super(ConsistencyLoss, self).__init__()
        self.pixpro_pos_ratio = args.pixpro_pos_ratio
        self.pixpro = PixPro(args)
        if args.data == 'endo18':
            self.class_num = 12
        elif args.data == 'cata':
            self.class_num = int(num_class_table[args.tag])
        print('Class_num:', self.class_num)

    def forward(self, im_1, im_2, im_3, im_4, im_5, im_6,mask_1, mask_2, mask_3, mask_4, mask_5,mask_6):

        pred_1, pred_2, proj_1_ng, proj_2_ng, proj_adj1_ng, proj_adj2_ng, proj_adj3_ng, proj_neg3_ng = self.pixpro(im_1, im_2, im_3, im_4, im_5,im_6)
        #print('pred shape:', pred_1.shape) #2,256,28,28
        #print('proj shape:', proj_1_ng.shape) #2,256,28,28

        N, C, H, W = pred_1.shape
        mask_1 = F.interpolate(mask_1, size=[H,W], mode='nearest')
        mask_2 = F.interpolate(mask_2, size=[H,W], mode='nearest')
        mask_3 = F.interpolate(mask_3, size=[H,W], mode='nearest')
        mask_4 = F.interpolate(mask_4, size=[H,W], mode='nearest')
        mask_5 = F.interpolate(mask_5, size=[H, W], mode='nearest')
        mask_6 = F.interpolate(mask_6, size=[H,W], mode='nearest') #mask = label


        loss = regression_loss(pred_1, proj_2_ng, proj_adj1_ng, proj_adj2_ng, proj_adj3_ng, proj_neg3_ng, mask_1, mask_2, mask_3, mask_4, mask_5, mask_6,self.class_num) \
            + regression_loss(pred_2, proj_1_ng, proj_adj1_ng, proj_adj2_ng, proj_adj3_ng, proj_neg3_ng, mask_2, mask_1, mask_3, mask_4, mask_5, mask_6,self.class_num)
        
        return loss


if __name__ == '__main__':
    import torch
    opt = parse_option(stage='pre-train')
    net = ConsistencyLoss(opt).cuda()
    img = torch.randn(2, 4, 3, 224, 224).cuda()
    mask = torch.randn(2,1,224,224).cuda()
    y= net(img,img,img,img,img,mask,mask,mask,mask,mask)

