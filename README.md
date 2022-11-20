# Decontamination_Transformer
PyTorch implementaton of our paper "Decontamination Transformer for Blind Image Inpainting".

In this paper, We propose a carefully-designed de-contamination model for the task of blind image inpainting.


## Installation and Environment
- Prerequisities: Python 3.6 & Pytorch (at least 1.4.0) 
- Clone this repo
```
git clone https://github.com/Lcy0307/Decontamination_Transformer.git
cd Decontamination_Transformer
```

- We provide a conda environment script, please run the following command after cloning our repo.
```
conda env create -f environment.yml
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

### Preprocessing
#### Take Places2 for example, you should first prepare the filelist.
##### Training
- "./data_load/places2_train.txt"
- "./data_load/places2_val.txt"
- "./data_load/imagenet_train.txt"
- "./data_load/imagenet_val.txt"
##### Testing
- "./data_load/places2_test.txt"
- "./data_load/imagenet_test.txt"

## Training
More instructions can be found in train.sh for other dataset. Our model is trained on an Single RTX 3090 GPU.
#### Generating free-form mask of validation set
We generate 1000 free-form mask for validation sets.
```
python preprocess_val_mask.py
```

### First stage
```
# MPN
python train_MPN.py --dataset places2 --output_dir mpn_places2_strokemask --iter 600001 --batch 16
# CPN & IIN
python train_CPN_IIN.py --dataset places2 --output_dir RRtrans_gtmask_8_places2_stroke --use_gtmask --batch 8 --iter 600001
```
    
### Second stage
Cotraining MPN, CPN and IIN.
```
# cotrain MPN & CPN & IIN
python train_cotrain.py --dataset places2 --inpaint_ckpt RRtrans_gtmask_8_places2_stroke/checkpoint.pt --mask_ckpt mpn_places2_strokemask/checkpoint.pt --output_dir RRtrans_cotrain_places2 --batch 8 --iter 200001
```

## Testing
More instructions can be found in test.sh for other dataset.
### Using the pre-trained models
- Download the [pre-trained models](https://drive.google.com/drive/folders/17ge5uhZM6QD9i37PUPTpVTLdyMgygVCQ?usp=sharing), here we provide the pre-trained models for the FFHQ, Places2, ImageNet datasets.
```
python test.py --dataset places2 --ckpt checkpoint/RRtrans_cotrain_all_loss_partial_noise_8_stroke_places2/checkpoint.pt
```

## Reproducibility
If you want to reproduce the experimental results of our FFHQ model on training and testing, please preprocess the data by preprocess_data.py.
We first resize FFHQ, CelebA, Imagenet images into 256x256 and save by PIL.
Places2 and ImageNet should be fine by using original data.
```
python preprocess_data.py
```


## Acknowledgments
Our code is based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).
- The implementation of the gated convolution is borrowed from [GatedConvolution_pytorch](https://github.com/avalonstrel/GatedConvolution_pytorch).
- The implementation of the intra/inter attention is borrowed from [Stripformer](https://github.com/pp00704831/Stripformer).
