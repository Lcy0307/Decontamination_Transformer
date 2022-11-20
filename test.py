import argparse
from base64 import decode, encode
import math
import random
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from dataset_test import Places2_strokemask, FFHQ_strokemask, ImageNet_strokemask
from mask_utils import ConfidenceDrivenMaskLayer
from metrics import lpips_measure, ssim, compute
from mpn_model_rename.architecture import MPN
from models.CPN_IIN import RR_inpaint_model_minus_trans




if __name__ == "__main__":
    device = "cuda:0"
    
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--output_dir",
        type=str,
        default='out',
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='./checkpoint/RRtrans_cotrain_all_loss_partial_noise_8_stroke_places2/checkpoint.pt',
    )
    parser.add_argument(
        "--save", action="store_true", help="save image"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()

    # CPN and IIN
    gatedconv = RR_inpaint_model_minus_trans().to(device)
    # MPN
    mask_model = MPN().to(device)

    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    mask_ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    gatedconv.load_state_dict(ckpt["gatedconv"])
    mask_model.load_state_dict(mask_ckpt["mask_model"])

    if args.dataset == "ffhq":
        main_dataset = FFHQ_strokemask()
    elif args.dataset == "places2":
        main_dataset = Places2_strokemask()
    elif args.dataset == "imagenet":
        main_dataset = ImageNet_strokemask()
    data_loader_test = torch.utils.data.DataLoader(
        main_dataset,
        batch_size=20,
        num_workers=10,
        shuffle=False
    )

    smoother = ConfidenceDrivenMaskLayer(iters=4).to(device)
    criterion = nn.MSELoss()
    lpips_model = lpips_measure(device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    gatedconv.eval()
    mask_model.eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []
    mse_list = []
    psnr_list_sharp = []
    ssim_list_sharp = []
    lpips_list_sharp = []
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
            predict_noise, predict_imgs = gatedconv(corrupt_imgs, predict_masks)
            
            # compute metrics
            psnr = compute(predict_imgs, gt)
            ssim_  =  ssim(predict_imgs, gt).mean().item()
            lpips = lpips_model.run(predict_imgs, gt).mean().item()
            mse = criterion(predict_masks, masks_smooth)

            print(data_iter_step, psnr)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim_)
            lpips_list.append(lpips)
            mse_list.append(mse.item() * 1000)

            if args.save:
                utils.save_image(
                            corrupt_imgs[0],
                            f"{str(output_dir)}/{data_iter_step}_corrupt_imgs.png",
                            nrow=int(4),
                            normalize=True,
                            range=(-1, 1),
                        )
                utils.save_image(
                            predict_imgs[0],
                            f"{str(output_dir)}/{data_iter_step}_predict_imgs.png",
                            nrow=int(4),
                            normalize=True,
                            range=(-1, 1),
                        )
                utils.save_image(
                            gt[0],
                            f"{str(output_dir)}/{data_iter_step}_gt.png",
                            nrow=int(4),
                            normalize=True,
                            range=(-1, 1),
                        )
                utils.save_image(
                            masks_smooth[0],
                            f"{str(output_dir)}/{data_iter_step}_masks.png",
                            nrow=int(4),
                            normalize=True,
                            range=(0, 1),
                        )
                utils.save_image(
                            noise[0],
                            f"{str(output_dir)}/{data_iter_step}_noise.png",
                            nrow=int(4),
                            normalize=True,
                            range=(-1, 1),
                        )
                utils.save_image(
                            predict_masks[0],
                            f"{str(output_dir)}/{data_iter_step}_predict_masks.png",
                            nrow=int(4),
                            normalize=True,
                            range=(0, 1),
                        )
        if data_iter_step > 50:
            break
            
    
    psnr_np = np.asarray(psnr_list)
    ssim_np = np.asarray(ssim_list)
    lpips_np = np.asarray(lpips_list)
    mse_np = np.asarray(mse_list)
    print("PSNR_SMOOTH:",f"{psnr_np.mean():.3f}")
    print("SSIM_SMOOTH:",f"{ssim_np.mean():.4f}")
    print("LPIPS_SMOOTH:",f"{lpips_np.mean():.4f}")
    print("MSE_SMOOTH:",f"{mse_np.mean():.3f}")



            




            