from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as trns
from torch.utils.data import Dataset

ffhq_filelist = './data_load/ffhq_test.txt'
celebA_filelist = './data_load/celebA_test.txt'
places2_filelist = './data_load/places2_test.txt'
imagenet_filelist = './data_load/imagenet_test.txt'
irregular_mask_folder = '/eva_data2/irregular_mask/test_mask/'

# from https://github.com/shepnerd/blindinpainting_vcnet
class VC_mask_generator():
    def __init__(self):
        self.parts=8
        self.maxBrushWidth=20
        self.maxLength=80
        self.maxVertex=16
        self.maxAngle=360

    def np_free_form_mask(self,maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)

            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        return mask

    def generate(self, H, W):
        mask = np.zeros((H, W, 1), np.float32)
        for i in range(self.parts):
            p = self.np_free_form_mask(self.maxVertex, self.maxLength, self.maxBrushWidth, self.maxAngle, H, W)
            mask = mask + p
        mask = np.minimum(mask, 1.0)
        return np.reshape(mask, (1, H, W))


class Places2_noiseffhq_strokemask(Dataset):
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
        ffhq_list = []
        with open(ffhq_filelist, 'r') as out:
            for line in out:
                ffhq_list.append(line.rstrip("\n"))
        self.ffhq_np = np.asarray(ffhq_list)
        self.noise_length = self.ffhq_np.shape[0]
        
        print("img, noise length:",self.places2_np.shape[0], self.ffhq_np.shape[0])
            
        self.mask_generator = VC_mask_generator()

    def __len__(self):
        return self.places2_np.shape[0]

    def __getitem__(self, index):
        rand = torch.randint(self.noise_length,(1,)).item()

        img = Image.open(self.places2_np[index]).convert('RGB')
        noise = Image.open(self.ffhq_np[rand]).convert('RGB')
        
        img = trns.CenterCrop(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.Resize(self.image_shape,trns.InterpolationMode.BICUBIC)(noise)
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = self.mask_generator.generate(self.image_shape[0], self.image_shape[1])
        mask = torch.from_numpy(mask)

        return img, noise, mask
    
class ImageNet_noisecolor_strokemask(Dataset):
    def __init__(self, image_shape = (256,256)):
        
        self.normalize = trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        self.image_shape = image_shape
        # imagenet
        imagenet_list = []
        with open(imagenet_filelist, 'r') as out:
            for line in out:
                imagenet_list.append(line.rstrip("\n"))
        self.imagenet_np = np.asarray(imagenet_list)
        
        self.mask_generator = VC_mask_generator()


    def __len__(self):
        return self.imagenet_np.shape[0]

    def __getitem__(self, index):
        img = Image.open(self.imagenet_np[index]).convert('RGB')
        
        img = trns.CenterCrop(self.image_shape)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        

        noise = torch.zeros_like(img)
        noise[0,...] = torch.rand(1)[0].item() * 2 - 1.
        noise[1,...] = torch.rand(1)[0].item() * 2 - 1.
        noise[2,...] = torch.rand(1)[0].item() * 2 - 1.

        mask = self.mask_generator.generate(self.image_shape[0], self.image_shape[1])
        mask = torch.from_numpy(mask)

        return img, noise, mask
    
class FFHQ_noisepc2_strokemask(Dataset):
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
        places2_list = []
        with open(places2_filelist, 'r') as out:
            for line in out:
                places2_list.append(line.rstrip("\n"))
        self.places2_np = np.asarray(places2_list)
        
        self.mask_generator = VC_mask_generator()


    def __len__(self):
        return self.ffhq_np.shape[0]

    def __getitem__(self, index):

        img = Image.open(self.ffhq_np[index]).convert('RGB')
            

        rand = torch.randint(self.places2_np.shape[0],(1,)).item()
        noise = Image.open(self.places2_np[rand]).convert('RGB')
        noise = trns.CenterCrop(self.image_shape)(noise)
        
        img = trns.Resize(self.image_shape,trns.InterpolationMode.BICUBIC)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = self.mask_generator.generate(self.image_shape[0], self.image_shape[1])
        mask = torch.from_numpy(mask)

        return img, noise, mask
    
class Places2_irregularmask(Dataset):
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
        self.noise_length = self.imagenet_np.shape[0]
        
        print("img, noise length:",self.places2_np.shape[0], self.imagenet_np.shape[0])

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

        mask = Image.open(irregular_mask_folder + f'{str(index).zfill(5)}.png').convert('L')
        mask = trns.ToTensor()(mask)

        return img, noise, mask
    
class ImageNet_irregularmask(Dataset):
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

        mask = Image.open(irregular_mask_folder + f'{str(index).zfill(5)}.png').convert('L')
        mask = trns.ToTensor()(mask)

        return img, noise, mask
    
class FFHQ_irregularmask(Dataset):
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


    def __len__(self):
        return self.ffhq_np.shape[0]

    def __getitem__(self, index):

        img = Image.open(self.ffhq_np[index]).convert('RGB')
            

        rand = torch.randint(self.celebA_np.shape[0],(1,)).item()
        noise = Image.open(self.celebA_np[rand]).convert('RGB')
        noise = trns.Resize(self.image_shape,trns.InterpolationMode.BICUBIC)(noise)
        
        img = trns.Resize(self.image_shape,trns.InterpolationMode.BICUBIC)(img)
        img = trns.RandomHorizontalFlip()(img)
        img = trns.ToTensor()(img)
        img = self.normalize(img)
        
        noise = trns.RandomHorizontalFlip()(noise)
        noise = trns.ToTensor()(noise)
        noise = self.normalize(noise)

        mask = Image.open(irregular_mask_folder + f'{str(index).zfill(5)}.png').convert('L')
        mask = trns.ToTensor()(mask)

        return img, noise, mask
    
    
