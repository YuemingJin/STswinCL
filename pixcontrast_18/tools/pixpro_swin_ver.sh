#!/bin/bash

set -e
set -x

data_dir="./data/imagenet/"
output_dir="./endo18/mswin_minput_ver_x"
pretrainpth='endo18/mswin_ver_x/ckpt/epoch_x_checkpoint.t7'

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port 1234 --nproc_per_node=1 \
    main_pretrain_swinv5.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    --data endo18 \
    --pretrainpth ${pretrainpth} \
    \
    --zip --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 4 \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 150 \
    --amp-opt-level O0 \
    \
    --save-freq 10 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 5.0 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 0. \
