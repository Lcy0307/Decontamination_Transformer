import numpy as np
import torch
import torch.nn as nn

from mask_utils import ConfidenceDrivenMaskLayer


def compute(pred, target):
    pred = torch.clip((pred * 0.5 + 0.5) * 255, 0., 255.)
    target = torch.clip((target * 0.5 + 0.5) * 255, 0., 255.)
    mse = torch.mean((pred-target) ** 2,dim=(1,2,3))
    return torch.mean(20 * torch.log10(255.0 / torch.sqrt(mse))).item()





# have noise
def validate_model_noise(mask_model, gatedconv, data_loader_test,device):

    smoother = ConfidenceDrivenMaskLayer().to(device)
    criterion = nn.MSELoss()

    gatedconv.eval()
    mask_model.eval()

    psnr_list = []
    mse_list = []
    psnr_list_sharp = []
    mse_list_sharp = []

    for data_iter_step, (gt, noise, masks) in enumerate(data_loader_test):
        with torch.no_grad():
            
            masks = masks.to(device)
            noise = noise.to(device)
            gt = gt.to(device)

            smooth_masks = smoother(1 - masks) + masks
            smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
            masks_smooth = smooth_masks
            corrupt_imgs = noise * masks_smooth + gt * (1. - masks_smooth)
            
            predict_masks = mask_model(corrupt_imgs)
            _, predict_imgs = gatedconv(corrupt_imgs, predict_masks)
            

            psnr = compute(predict_imgs, gt)
            mse = criterion(predict_masks, masks_smooth)
            
            psnr_list.append(psnr)
            mse_list.append(mse.item() * 1000)
            


    for data_iter_step, (gt, noise, masks) in enumerate(data_loader_test):
        with torch.no_grad():
            
            masks = masks.to(device)
            noise = noise.to(device)
            gt = gt.to(device)

            corrupt_imgs = noise * masks + gt * (1. - masks)

        
            predict_masks = mask_model(corrupt_imgs)
            _, predict_imgs = gatedconv(corrupt_imgs, predict_masks)
            
            
            psnr = compute(predict_imgs, gt)
            mse = criterion(predict_masks, masks)
            mse_list_sharp.append(mse.item() * 1e3)
            
            psnr_list_sharp.append(psnr)
    psnr_np = np.asarray(psnr_list)
    mse_np = np.asarray(mse_list)
    psnr_np_sharp = np.asarray(psnr_list_sharp)
    mse_np_sharp = np.asarray(mse_list_sharp)
    
    return psnr_np.mean(), mse_np.mean(), psnr_np_sharp.mean(), mse_np_sharp.mean()
    

            




            