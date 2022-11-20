# Decontamination_Transformer
PyTorch implementaton of our paper "Decontamination Transformer for Blind Image Inpainting".

In this paper, We propose a carefully-designed de-contamination model for the task of blind image inpainting.


## Installation and Environment
- Prerequisities: Python 3.7 & Pytorch (at least 1.4.0) 
- Clone this repo
```
git clone https://github.com/Lcy0307/Decontamination_Transformer.git
cd Decontamination_Transformer
```

- We provide a conda environment script, please run the following command after cloning our repo.
```
conda env create -f vqi2i_env.yml
```
## Datasets
- FFHQ dataset: You can follow the instructions in FFHQ [website](https://github.com/NVlabs/ffhq-dataset) to download the FFHQ dataset.
- CelebA-HQ dataset: You can follow the instructions in Large-scale CelebFaces Attributes [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to download the CelebA dataset.
- Places2 dataset: You can follow the instructions in Places2 [website](http://places2.csail.mit.edu/download.html) to download the Places2(training and val) dataset.
- ImageNet dataset: You can follow the instructions in ImageNet [website](https://www.image-net.org/) to download the ImageNet(training and val) dataset.

> Please generate the file list in this format(absolute path) and name it like "./data_load/ffhq_train.txt":
```
/home/lcy/dset/ffhq256x256/25870.png
/home/lcy/dset/ffhq256x256/25871.png
/home/lcy/dset/ffhq256x256/25871.png
...
```

## First Stage of training
### Take Places2 for example, you should first prepare the filelist.
- "./data_load/places2_train.txt"
- "./data_load/places2_val.txt"
- "./data_load/imagenet_train.txt"
- "./data_load/imagenet_val.txt"
```
# MPN
python train_MPN.py --dataset places2 --output_dir mpn_places2_strokemask --iter 600001 --batch 16
# CPN & IIN
python train_CPN_IIN.py --dataset places2 --output_dir RRtrans_gtmask_8_places2_stroke --use_gtmask --batch 8 --iter 600001
```
    
## Second stage of training
```
# cotrain MPN & CPN & IIN
python train_cotrain.py --dataset places2 --inpaint_ckpt RRtrans_gtmask_8_places2_stroke/checkpoint.pt --mask_ckpt mpn_places2_strokemask/checkpoint.pt --output_dir RRtrans_cotrain_places2 --batch 8 --iter 200001
```


## Acknowledgments
Our code is based on [VQGAN](https://github.com/CompVis/taming-transformers).
The implementation of the disentanglement architecture is borrowed from [MUNIT](https://github.com/NVlabs/MUNIT).
