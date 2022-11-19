import math
import numbers
import torch
import numpy as np
import scipy.stats as st
import cv2

from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
import skimage.morphology as sm


# https://github.com/shepnerd/blindinpainting_vcnet
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
            brushWidth = np.random.randint(2, maxBrushWidth + 1) // 2 * 2
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


# https://github.com/birdortyedi/vcnet-blind-image-inpainting
def gauss_kernel(size=15, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size

    x = np.linspace(-sigma - interval / 2, sigma + interval / 2, size + 1)

    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()

    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    out_filter = np.tile(out_filter, [outchannels, inchannels, 1, 1])

    return out_filter


# https://github.com/birdortyedi/vcnet-blind-image-inpainting
class GaussianBlurLayer(nn.Module):
    def __init__(self, size, sigma, in_channels=1, stride=1, pad=1):
        super(GaussianBlurLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.ch = in_channels
        self.stride = stride
        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x):
        kernel = gauss_kernel(self.size, self.sigma, self.ch, self.ch)
        kernel_tensor = torch.from_numpy(kernel)
        kernel_tensor = kernel_tensor.cuda()
        x = self.pad(x)
        blurred = F.conv2d(x, kernel_tensor, stride=self.stride)
        return blurred

# https://github.com/birdortyedi/vcnet-blind-image-inpainting
class ConfidenceDrivenMaskLayer(nn.Module):
    def __init__(self, size=15, sigma=1.0 / 40, iters=4):
        super(ConfidenceDrivenMaskLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.iters = iters
        self.propagation_layer = GaussianBlurLayer(size, sigma, pad=size // 2)

    def forward(self, mask):
        # here mask 1 indicates missing pixels and 0 indicates the valid pixels
        init = 1 - mask
        mask_confidence = None
        for i in range(self.iters):
            mask_confidence = self.propagation_layer(init)
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
        return mask_confidence

def kernel_tensor_generate():
    sigma_list = np.linspace(0.0, 3.0, num=401)
    sigma_list = sigma_list[1:]

    kernel_list = []
    kernel_size = [3, 5, 7, 11, 13, 15]
    for i in range(sigma_list.size):
        np_kernel = torch.from_numpy(gauss_kernel(size=15, sigma=sigma_list[i], inchannels=1, outchannels=1))
        kernel_list.append(np_kernel.cuda())
    
    return kernel_list

class GaussianBlurLayer_input_kernel(nn.Module):
    def __init__(self, stride, pad):
        super(GaussianBlurLayer_input_kernel, self).__init__()
        self.stride = stride
        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x, kernel_tensor):
        x = self.pad(x)
        blurred = F.conv2d(x, kernel_tensor, stride=self.stride)
        return blurred

class ConfidenceDrivenMaskLayer_input_iter(nn.Module):
    def __init__(self, size=15):
        super(ConfidenceDrivenMaskLayer_input_iter, self).__init__()
        self.size = size
        self.propagation_layer = GaussianBlurLayer_input_kernel(1,pad=size // 2)
        self.kernel_lt = kernel_tensor_generate()
        self.kernel_num = len(self.kernel_lt)
        print("Kernel number:", self.kernel_num)

    def forward(self, mask):
        # here mask 1 indicates missing pixels and 0 indicates the valid pixels
        iter_rand = torch.randint(7,(1,)).item()
        #kernel_rand = torch.randint(self.kernel_num,(1,)).item()
        init = 1 - mask
        mask_confidence = 0.
        for i in range(iter_rand):
            kernel_rand = torch.randint(self.kernel_num,(1,)).item()
            mask_confidence = self.propagation_layer(init, self.kernel_lt[kernel_rand])
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
        return mask_confidence
    
