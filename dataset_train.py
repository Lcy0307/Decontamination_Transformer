from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as trns
from torch.utils.data import Dataset
from mask_utils import ConfidenceDrivenMaskLayer_input_iter,VC_mask_generator

ffhq_filelist = './data_load/ffhq_train.txt'
celebA_filelist = './data_load/celebA_train.txt'
places2_filelist = './data_load/places2_train.txt'
imagenet_filelist = './data_load/imagenet_train.txt'

ffhq_val_filelist = './data_load/ffhq_val.txt'
celebA_val_filelist = './data_load/celebA_val.txt'
places2_val_filelist = './data_load/places2_val.txt'
imagenet_val_filelist = './data_load/imagenet_val.txt'

# Training dataset
    
class ImageNet_strokemask(Dataset):
    def __init__(self, image_shape = (256,256)):
        
        self.normalize = trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        self.image_shape = image_shape
        # places2 dataset length
        place2_list = []
        with open(places2_filelist, 'r') as out:
            for line in out:
                place2_list.append(line.rstrip("\n"))
        self.places2_np = np.asarray(place2_list)
        # imagenet
        imagenet_list = []
        with open(imagenet_filelist, 'r') as out:
            for line in out:
                imagenet_list.append(line.rstrip("\n"))
        self.imagenet_np = np.asarray(imagenet_list)
        self.noise_length = self.places2_np.shape[0]
        
        self.mask_generator = VC_mask_generator()


    def __len__(self):
        return self.imagenet_np.shape[0]

    def __getitem__(self, index):
        rand = torch.randint(self.noise_length,(1,)).item()

        noise = Image.open(self.places2_np[rand]).convert('RGB')
        img = Image.open(self.imagenet_np[index]).convert('RGB')
        
        img = trns.CenterCrop(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.CenterCrop(self.image_shape)(noise)
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = self.mask_generator.generate(self.image_shape[0], self.image_shape[1])
        mask = torch.from_numpy(mask)

        return img, noise, mask

    
class Places2_strokemask(Dataset):
    def __init__(self,image_shape = (256,256)):
        
        self.normalize = trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.image_shape = image_shape
        # places2 dataset length
        place2_list = []
        with open(places2_filelist, 'r') as out:
            for line in out:
                place2_list.append(line.rstrip("\n"))
        self.places2_np = np.asarray(place2_list)
        # imagenet
        imagenet_list = []
        with open(imagenet_filelist, 'r') as out:
            for line in out:
                imagenet_list.append(line.rstrip("\n"))
        self.imagenet_np = np.asarray(imagenet_list)
        self.noise_length = self.imagenet_np.shape[0]
        
        print("img, noise length:",self.places2_np.shape[0], self.imagenet_np.shape[0])
            
        self.mask_generator = VC_mask_generator()

    def __len__(self):
        return self.places2_np.shape[0]

    def __getitem__(self, index):
        rand = torch.randint(self.noise_length,(1,)).item()

        img = Image.open(self.places2_np[index]).convert('RGB')
        noise = Image.open(self.imagenet_np[rand]).convert('RGB')
        
        img = trns.CenterCrop(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.CenterCrop(self.image_shape)(noise)
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = self.mask_generator.generate(self.image_shape[0], self.image_shape[1])
        mask = torch.from_numpy(mask)

        return img, noise, mask
                
class FFHQ_strokemask(Dataset):
    def __init__(self, image_shape = (256,256)):
        
        self.normalize = trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.image_shape = image_shape
        # ffhq dataset length
        ffhq_list = []
        with open(ffhq_filelist, 'r') as out:
            for line in out:
                ffhq_list.append(line.rstrip("\n"))
        self.ffhq_np = np.asarray(ffhq_list)
        #
        celebA_list = []
        with open(celebA_filelist, 'r') as out:
            for line in out:
                celebA_list.append(line.rstrip("\n"))
        self.celebA_np = np.asarray(celebA_list)
        # imagenet
        imagenet_list = []
        with open(imagenet_filelist, 'r') as out:
            for line in out:
                imagenet_list.append(line.rstrip("\n"))
        self.imagenet_np = np.asarray(imagenet_list)
        
        print("img, noiseA, noiseB length",self.ffhq_np.shape[0], self.celebA_np.shape[0],self.imagenet_np.shape[0])
        self.mask_generator = VC_mask_generator()


    def __len__(self):
        return self.ffhq_np.shape[0]

    def __getitem__(self, index):
        rand_ = torch.rand(1)[0].item()

        img = Image.open(self.ffhq_np[index]).convert('RGB')
        if rand_ > 0.5:
            rand = torch.randint(self.imagenet_np.shape[0],(1,)).item()
            noise = Image.open(self.imagenet_np[rand]).convert('RGB')
            noise = trns.CenterCrop(self.image_shape)(noise)
            
        else:
            rand = torch.randint(self.celebA_np.shape[0],(1,)).item()
            noise = Image.open(self.celebA_np[rand]).convert('RGB')
            noise = trns.Resize(self.image_shape,trns.InterpolationMode.BICUBIC)(noise)
        
        img = trns.Resize(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = self.mask_generator.generate(self.image_shape[0], self.image_shape[1])
        mask = torch.from_numpy(mask)

        return img, noise, mask
    
    
        
# You can choose to use validation set or not
class ImageNet_val(Dataset):
    def __init__(self, image_shape = (256,256)):
        
        self.normalize = trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        self.image_shape = image_shape
        # places2 dataset length
        place2_list = []
        with open(places2_val_filelist, 'r') as out:
            for line in out:
                place2_list.append(line.rstrip("\n"))
        self.places2_np = np.asarray(place2_list)
        # imagenet
        imagenet_list = []
        with open(imagenet_val_filelist, 'r') as out:
            for line in out:
                imagenet_list.append(line.rstrip("\n"))
        self.imagenet_np = np.asarray(imagenet_list)
        self.noise_length = self.places2_np.shape[0]


    def __len__(self):
        return 1000
        #return self.imagenet_np.shape[0]

    def __getitem__(self, index):

        noise = Image.open(self.places2_np[index]).convert('RGB')
        img = Image.open(self.imagenet_np[index]).convert('RGB')
        
        img = trns.CenterCrop(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.CenterCrop(self.image_shape)(noise)
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)
        
        mask = Image.open(f'val_mask/binary_mask_{str(index).zfill(3)}.png').convert('L')
        mask = trns.ToTensor()(mask)


        return img, noise, mask
    
    
class Places2_val(Dataset):
    def __init__(self,image_shape = (256,256)):
        
        self.normalize = trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        self.image_shape = image_shape
        # places2 dataset length
        place2_list = []
        with open(places2_val_filelist, 'r') as out:
            for line in out:
                place2_list.append(line.rstrip("\n"))
        self.places2_np = np.asarray(place2_list)
        # imagenet
        imagenet_list = []
        with open(imagenet_val_filelist, 'r') as out:
            for line in out:
                imagenet_list.append(line.rstrip("\n"))
        self.imagenet_np = np.asarray(imagenet_list)
        self.noise_length = self.imagenet_np.shape[0]


    def __len__(self):
        return self.places2_np.shape[0]

    def __getitem__(self, index):

        noise = Image.open(self.places2_np[index]).convert('RGB')
        img = Image.open(self.imagenet_np[index]).convert('RGB')
        
        img = trns.CenterCrop(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.CenterCrop(self.image_shape)(noise)
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = Image.open(f'val_mask/binary_mask_{str(index).zfill(3)}.png').convert('L')
        mask = trns.ToTensor()(mask)

        return img, noise, mask
    

class FFHQ_val(Dataset):
    def __init__(self, image_shape = (256,256)):
        
        self.normalize = trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.image_shape = image_shape
        # ffhq dataset length
        ffhq_list = []
        with open(ffhq_val_filelist, 'r') as out:
            for line in out:
                ffhq_list.append(line.rstrip("\n"))
        self.ffhq_np = np.asarray(ffhq_list)
        #
        celebA_list = []
        with open(celebA_val_filelist, 'r') as out:
            for line in out:
                celebA_list.append(line.rstrip("\n"))
        self.celebA_np = np.asarray(celebA_list)
        
        
    def __len__(self):
        return self.ffhq_np.shape[0]

    def __getitem__(self, index):
        rand_ = torch.rand(1)[0].item()

        img = Image.open(self.ffhq_np[index]).convert('RGB')
        noise = Image.open(self.celebA_np[index]).convert('RGB')
        noise = trns.Resize(self.image_shape,trns.InterpolationMode.BICUBIC)(noise)
        
        img = trns.Resize(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = Image.open(f'val_mask/binary_mask_{str(index).zfill(3)}.png').convert('L')
        mask = trns.ToTensor()(mask)

        return img, noise, mask
