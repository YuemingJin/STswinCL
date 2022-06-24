import os

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform


def get_loader(args):
    transform = get_transform(args.crop, args.image_size_h, args.image_size_w)
    if args.data == 'endo18':
        from .dataset import pretrainDataset
        train_dataset = pretrainDataset(transform=transform)
    elif args.data == 'cata':
        from .dataset_cata import pretrainDataset
        train_dataset = pretrainDataset(transform=transform,tag=args.tag)

    # sampler
    indices = np.arange(dist.get_rank(), len(train_dataset), dist.get_world_size())
    if args.zip and args.cache_mode == 'part':
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSampler(train_dataset)

    # dataloader
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True)
