#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def DiceLoss( predict, target):
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)
    inter = torch.sum(predict*target)
    union = torch.sum(predict)+torch.sum(target)+1e-6
    return 1 - 2*inter/union
    
class OhemCELoss2D(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self,
                 n_min,
                 thresh=0.7,
                 ignore_index=-1):

        super(OhemCELoss2D, self).__init__(None, None, ignore_index, reduction='none')

        self.thresh = -math.log(thresh)
        self.n_min = n_min #最少要计算多少个pixel的loss
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return self.OhemCELoss(pred, target)

    def OhemCELoss(self, logits, labels):
        loss = super(OhemCELoss2D, self).forward(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

    
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

# class BCELoss(nn.Module):
#     def __init__(self):
#         super(BCELoss, self).__init__()
#     def forward(self, predict, target):
#         predict = F.softmax(predict,dim=1)
#         return F.binary_cross_entropy(predict, target)
    
class BinaryCrossEntropyLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        return F.binary_cross_entropy(predict, target)

class BCELoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(BCELoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryCrossEntropyLoss(**self.kwargs)
        total_loss = []
        predict = F.softmax(predict, dim=1)
        avg_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    loss *= self.weights[i]
                total_loss.append(loss)
                avg_loss += loss
        return avg_loss/target.shape[1]