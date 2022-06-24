# Exploring Intra- and Inter-Video Relation for Surgical Semantic Scene Segmentation
by [Yueming Jin](https://yuemingjin.github.io/), Yang Yu, [Cheng Chen](https://scholar.google.com.hk/citations?user=bRe3FlcAAAAJ&hl=en), [Zixu Zhao](https://scholar.google.com.hk/citations?user=GSQY0CEAAAAJ&hl=zh-CN), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/), [Danail Stoyanov](https://www.ucl.ac.uk/surgical-robot-vision/).

## Introduction
* The Pytorch implementation for our paper '[Exploring Intra- and Inter-Video Relation for Surgical Semantic Scene Segmentation](https://arxiv.org/abs/2203.15251)', accepted at IEEE Transactions on Medical Imaging (TMI), 2022.

<p align="center">
  <img src="figure/overview.png"  width="800"/>
</p>


## Dataset

* We use the dataset [EndoVis18](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Home/) and [CaDIS](https://ts-media-content.s3-eu-west-1.amazonaws.com/machine-learning/datasets/CaDISv2.zip).

## Results

* More visual results can be found in this [video](https://yuemingjin.github.io/video/2022TMI.mp4)

## Usage

* Check dependencies:
   ```
   - pytorch 1.8.0
   - opencv-python
   - tqdm
   - timm
   - pi
   - numpy
   - sklearn
   ```

* Clone this repo
    ```shell
    git clone https://github.com/YuemingJin/STswinCL
    ```

* Training process

1. Training Transformer based segmentation model (Intra-video)

* Switch folder ``$ cd ./seg18/``

* Use ``$ python train_swin.py`` to start the training; parameter setting and training script refer to ``exp.sh``

2. Training Contrastive model (Inter-video)

* Switch folder ``$ cd ./pixcontrast_18/``

* Use ``$ sh tools/pixpro_swin_ver.sh`` to start the training.

3. Fine-tuning the segmentation model (Joint Intra and Inter)

* Switch folder ``$ cd ./seg18/``

* Use ``$ python train_CL_ft_mswin_sgd_minput.py`` to start the training; parameter setting and training script refer to ``exp.sh``


## Test & Visualization

* Use ``$ python test.py`` to test; parameter setting and script can refer to ``exp.sh``



### Note: 
seg18 and pixcontrast_18 are for EndoVis18; segcata and pixcontrast_cata are for CaDIS.
Here, we take EndoVis18 as the example. The usage for CaDIS is similar.




## Citation
If this repository is useful for your research, please consider citing:
```
@ARTICLE{9779714,
  author={Jin, Yueming and Yu, Yang and Chen, Cheng and Zhao, Zixu and Heng, Pheng-Ann and Stoyanov, Danail},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Exploring Intra- and Inter-Video Relation for Surgical Semantic Scene Segmentation}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3177077}
}
```

### Questions

For further question about the code or paper, please contact 'ymjin5341@gmail.com'


