import argparse
from base64 import decode, encode
import dis
import math
import random
import os
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
from mask_utils import ConfidenceDrivenMaskLayer_input_iter
from validate import validate_model_noise
from loss import Normalization, PerceptualLoss

try:
    import wandb

except ImportError:
    wandb = None


from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def compute(pred, target):
    pred = torch.clip((pred * 0.5 + 0.5) * 255, 0., 255.).int().float()
    target = torch.clip((target * 0.5 + 0.5) * 255, 0., 255.).int().float()
    mse = torch.mean((pred-target) ** 2,dim=(1,2,3))
    return torch.mean(20 * torch.log10(255.0 / torch.sqrt(mse))).item()

def compute_masked_region(pred, target, mask):
    pred = torch.clip((pred * 0.5 + 0.5) * 255, 0., 255.)
    target = torch.clip((target * 0.5 + 0.5) * 255, 0., 255.)
    mse = torch.sum((pred * mask-target * mask) ** 2,dim=(1,2,3)) / mask.sum(dim=(1,2,3))
    return torch.mean(20 * torch.log10(255.0 / torch.sqrt(mse))).item()


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean(), fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def train(args, main_loader, mask_model, gatedconv, discriminator, mask_model_optim, gatedconv_optim, optimizer_discriminator, val_dataset, device):
    main_loader = sample_data(main_loader)
    
    smoother = ConfidenceDrivenMaskLayer_input_iter().to(device)
    # Perceptual loss launch
    NormalVGG = Normalization(device)
    perceptual = PerceptualLoss().to(device)
    # best recorded
    best_psnr_smooth = psnr_smooth = 0.
    best_psnr_sharp = psnr_sharp = 0.
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}

    if args.distributed:
        gatedconv_module = gatedconv.module
        mask_model_module = mask_model

    else:
        gatedconv_module = gatedconv
        discriminator_module = discriminator
        mask_model_module = mask_model


    for idx in pbar:
        epoch = 0
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        imgs, blend_imgs, masks= next(main_loader)
        
        imgs = imgs.to(device)
        blend_imgs = blend_imgs.to(device)
        # mask shape (B,1,H,W)
        masks = masks.to(device)
        smooth_masks = torch.rand_like(masks,device=device)

        with torch.no_grad():
            # Quantization
            masks[masks > 0.5] = 1.
            masks[masks < 0.5] = 0.
            
            for j in range(imgs.shape[0]):
                smooth_masks[j:j+1] = masks[j:j+1] + smoother(1. - masks[j:j+1])
                
            smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
            corrupt_img = blend_imgs * smooth_masks + imgs * (1.- smooth_masks)
            
            image_masks = smooth_masks.clone()
            image_masks[image_masks < 1.] = 0.
            image_masks = 1. - image_masks
            
        gatedconv.train()
        discriminator.train()
        mask_model.train()
        
        # Discriminator Network
        requires_grad(mask_model, False)
        requires_grad(gatedconv, False)
        requires_grad(discriminator, True)
        optimizer_discriminator.zero_grad()
        
        with torch.no_grad():
            predict_masks = mask_model(corrupt_img.detach())
            _, predict_imgs = gatedconv(corrupt_img.detach(), predict_masks)
            output = predict_imgs * smooth_masks + imgs * (1. - smooth_masks)

        real_validity = discriminator(imgs.detach())
        fake_validity = discriminator(output.detach())
        # wgan gp failed
        # gp = compute_gradient_penalty(discriminator, output, imgs,device=device)

        real_d_loss, fake_d_loss = d_logistic_loss(real_validity, fake_validity)
        d_loss =  real_d_loss + fake_d_loss
        d_loss.backward()
        optimizer_discriminator.step()
        
        
        # Generator Network
        requires_grad(mask_model, True)
        requires_grad(gatedconv, True)
        requires_grad(discriminator, False)
        
        predict_masks = mask_model(corrupt_img.detach())
        predict_noise, predict_imgs = gatedconv(corrupt_img.detach(), predict_masks)
        
        # Mask loss
        criterion_mask = nn.MSELoss()
        predict_mask_loss = criterion_mask(predict_masks,smooth_masks.detach())
        
        # L1 loss
        criterion = nn.L1Loss()
        predict_l1_loss = criterion(predict_imgs, imgs.detach())
        predict_noise_loss = criterion(predict_noise * image_masks, blend_imgs.detach() * image_masks)
        #predict_noise_loss = criterion(predict_noise, blend_imgs.detach())
        
        # VGG loss
        predict_vgg_feature = NormalVGG(predict_imgs * 0.5 + 0.5)
        gt_vgg_feature = NormalVGG(imgs.detach() * 0.5 + 0.5)
        
        predict_vgg_loss = perceptual(predict_vgg_feature, gt_vgg_feature)
        
        # GAN loss
        output = predict_imgs * smooth_masks + imgs * (1.- smooth_masks)
        #adv_loss = -discriminator(output).mean()
        adv_loss = g_nonsaturating_loss(discriminator(output))

        # predict mask loss
        total_loss = 2.0 * predict_mask_loss + 1.0 * predict_l1_loss + 1.0 * predict_noise_loss\
            + 0.1 * predict_vgg_loss + 0.001 * adv_loss

        loss_dict["predict_mask"] = predict_mask_loss
        loss_dict["predict_l1"] = predict_l1_loss
        loss_dict["predict_vgg"] = predict_vgg_loss
        loss_dict["predict_noise"] = predict_noise_loss
        loss_dict["adv_g"] =  adv_loss
        loss_dict["real_d"] = real_d_loss
        loss_dict["fake_d"] = fake_d_loss
        loss_dict["d_loss"] = d_loss

        mask_model_optim.zero_grad()
        gatedconv_optim.zero_grad()
        total_loss.backward()
        gatedconv_optim.step()
        mask_model_optim.step()
        
        loss_reduced = reduce_loss_dict(loss_dict)

        predict_mask_val = loss_reduced["predict_mask"].mean().item()
        predict_l1_val = loss_reduced["predict_l1"].mean().item()
        predict_vgg_val = loss_reduced["predict_vgg"].mean().item()
        predict_noise_val = loss_reduced["predict_noise"].mean().item()
        adv_g_val = loss_reduced["adv_g"].mean().item()
        real_d_val = loss_reduced["real_d"].mean().item()
        fake_d_val = loss_reduced["fake_d"].mean().item()
        d_loss_val = loss_reduced["d_loss"].mean().item()

        if get_rank() == 0:
            if i % 1000 == 0:
                psnr_smooth, mse_smooth, psnr_sharp, mse_sharp = validate_model_noise(mask_model, gatedconv, val_dataset, device)
                if psnr_smooth > best_psnr_smooth and psnr_sharp > best_psnr_sharp:
                    best_psnr_smooth = psnr_smooth
                    best_psnr_sharp = psnr_sharp
                    print(f"smooth: {best_psnr_smooth}, sharp: {best_psnr_sharp} Saved!")
                    torch.save(
                        {
                            "mask_model": mask_model_module.state_dict(),
                            "discriminator": discriminator_module.state_dict(),
                            "gatedconv": gatedconv_module.state_dict(),
                            "args": args
                        },
                        f"{str(output_dir)}/checkpoint.pt",
                    )
                utils.save_image(
                        predict_masks,
                        f"{str(output_dir)}/predict_masks.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(0, 1),
                    )
                utils.save_image(
                        smooth_masks,
                        f"{str(output_dir)}/masks.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(0, 1),
                    )
                utils.save_image(
                        corrupt_img,
                        f"{str(output_dir)}/corrupt_img.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                utils.save_image(
                        output,
                        f"{str(output_dir)}/complete.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                utils.save_image(
                        predict_imgs,
                        f"{str(output_dir)}/predict.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                utils.save_image(
                        imgs,
                        f"{str(output_dir)}/imgs.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(-1, 1),
                    )
            pbar.set_description(
                (
                    f"mask: {predict_mask_val:.4f};"
                    f"l1: {predict_l1_val:.2f};"
                    f"vgg: {predict_vgg_val:.2f};"
                    f"real_d: {real_d_val:.2f};"
                    f"fake_d: {fake_d_val:.2f};"
                    #f"d_ls: {d_loss_val:.2f};"
                    #f"adv_g: {adv_g_val:.2f};"
                    # f"psnr_smooth: {psnr_smooth:.4f};"
                    # f"psnr_sharp: {psnr_sharp:.4f};"
                )
            )
            if wandb and args.wandb:
                wandb.log(
                    {
                        "Predict_mask": predict_mask_loss,
                        "Predict_l1": predict_l1_val,
                        "Predict_VGG": predict_vgg_val,
                        "Predict_adv": adv_g_val,
                        "mse_smooth": mse_smooth,
                        "mse_sharp": mse_sharp,
                        "psnr_smooth": psnr_smooth,
                        "psnr_sharp": psnr_sharp,
                        "real_d": real_d_val,
                        "fake_d": fake_d_val,
                        "d_loss": d_loss_val
                    }
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Cotrain_gated trainer")
    parser.add_argument('--arch', type=str, default='Gated')
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--iter", type=int, default=150001, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=16,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--inpaint_ckpt",
        type=str,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--mask_ckpt",
        type=str,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--threshold", type=float, help="learning rate")
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0

    gatedconv = None
    gatedconv_optim = None
    mask_model = None
    val_list = []

    from models.CPN_IIN import RR_inpaint_model_minus_trans
    from mpn_model_rename.architecture import Discriminator, MPN

    mask_model = MPN().to(device)
    gatedconv = RR_inpaint_model_minus_trans().to(device)
    discriminator = Discriminator().to(device)

    mask_model_optim = optim.Adam(
        mask_model.parameters(),
        lr=args.lr
    )
    
    gatedconv_optim = optim.Adam(
        gatedconv.parameters(),
        lr=args.lr
    )
    
    discriminator_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr
    )
    
    if args.inpaint_ckpt is not None:
        print("load model:", args.inpaint_ckpt)

        ckpt = torch.load(args.inpaint_ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.inpaint_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        gatedconv.load_state_dict(ckpt["gatedconv"])
        discriminator.load_state_dict(ckpt["discriminator"])

    if args.mask_ckpt is not None:
        print("load model:", args.mask_ckpt)

        ckpt = torch.load(args.mask_ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.mask_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        mask_model.load_state_dict(ckpt["mask_model"])
        

    if args.distributed:
        gatedconv = nn.parallel.DistributedDataParallel(
            gatedconv,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        mask_model = nn.parallel.DistributedDataParallel(
            mask_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)


    from dataset_train import Places2_strokemask, FFHQ_strokemask, ImageNet_strokemask
    from dataset_train import Places2_val, FFHQ_val, ImageNet_val

    if args.dataset == "ffhq":
        main_dataset = FFHQ_strokemask()
        val_dataset = FFHQ_val()
    elif args.dataset == "places2":
        main_dataset = Places2_strokemask()
        val_dataset = Places2_val()
    elif args.dataset == "imagenet":
        main_dataset = ImageNet_strokemask()
        val_dataset = ImageNet_val()
    
    main_loader = data.DataLoader(
        main_dataset,
        batch_size=args.batch,
        sampler=data_sampler(main_dataset, shuffle=True, distributed=args.distributed),
        num_workers=10,
        drop_last=True,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=20,
        num_workers=10,
        shuffle=False
    )

    if get_rank() == 0:
        if wandb is not None and args.wandb:
            wandb.init(project="Cotrain + gan + newval test")

    train(args, main_loader,mask_model, gatedconv, discriminator, mask_model_optim, gatedconv_optim, discriminator_optim, data_loader_test,device)
