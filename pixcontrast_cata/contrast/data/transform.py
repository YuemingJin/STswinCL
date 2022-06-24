import numpy as np
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from .rand_augment import rand_augment_transform
from . import transform_coord
import random


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



MEAN = [0.40789654, 0.44719302, 0.47026115]
STD = [0.28863828, 0.27408164, 0.27809835]

def get_transform(crop, image_size_h=256,image_size_w=448):
    normalize = transforms.Normalize(mean=[0.40789654, 0.44719302, 0.47026115], std=[0.28863828, 0.27408164, 0.27809835])

    transform_1 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size_h,image_size_w),
        transform_coord.RandomHorizontalFlipCoord(),
        #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur()], p=1.0),
        transforms.ToTensor(),
        normalize,
    ])
    transform_2 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size_h,image_size_w),
        transform_coord.RandomHorizontalFlipCoord(),
        #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur()], p=1.0),
        # transforms.RandomApply([ImageOps.solarize], p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    transform_3 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size_h,image_size_w),
        transform_coord.RandomHorizontalFlipCoord(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur()], p=0.1),
        # transforms.RandomApply([ImageOps.solarize], p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    transform_4 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size_h,image_size_w),
        transform_coord.RandomHorizontalFlipCoord(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur()], p=0.1),
        # transforms.RandomApply([ImageOps.solarize], p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    transform_5 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size_h,image_size_w),
        transform_coord.RandomHorizontalFlipCoord(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur()], p=0.1),
        # transforms.RandomApply([ImageOps.solarize], p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    transform_6 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size_h,image_size_w),
        transform_coord.RandomHorizontalFlipCoord(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur()], p=0.1),
        # transforms.RandomApply([ImageOps.solarize], p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


    transform = (transform_1, transform_2, transform_3, transform_4, transform_5, transform_6)

    return transform
