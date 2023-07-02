import argparse
from base64 import decode, encode
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


def train(args, main_loader, gatedconv, discriminator, gatedconv_optim, optimizer_discriminator, device):
    main_loader = sample_data(main_loader)
    
    smoother = ConfidenceDrivenMaskLayer_input_iter().to(device)
    # Perceptual loss launch
    NormalVGG = Normalization(device)
    perceptual = PerceptualLoss().to(device)
    psnr_smooth = 0.
    psnr_sharp = 0.
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}

    # if args.distributed:
    #     gatedconv_module = gatedconv.module

    # else:
    gatedconv_module = gatedconv
    discriminator_module = discriminator


    gatedconv.train()
    discriminator.train()

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
        
        # Discriminator Network
        requires_grad(gatedconv, False)
        requires_grad(discriminator, True)
        optimizer_discriminator.zero_grad()
        
        with torch.no_grad():
            if args.use_gtmask is True:
                # input smooth masks
                _, predict_imgs = gatedconv(corrupt_img.detach(), smooth_masks.detach())
            else:
                predict_imgs = gatedconv(corrupt_img.detach())
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
        requires_grad(gatedconv, True)
        requires_grad(discriminator, False)
        
        if args.use_gtmask is True:
            # input smooth masks
            predict_noise, predict_imgs = gatedconv(corrupt_img.detach(), smooth_masks.detach())
        else:
            predict_imgs = gatedconv(corrupt_img.detach())
        
        # L1 loss
        criterion = nn.L1Loss()
        predict_l1_loss = criterion(predict_imgs, imgs.detach())
        predict_noise_loss = criterion(predict_noise * image_masks, blend_imgs.detach() * image_masks)
        
        # VGG loss
        predict_vgg_feature = NormalVGG(predict_imgs * 0.5 + 0.5)
        gt_vgg_feature = NormalVGG(imgs.detach() * 0.5 + 0.5)
        
        predict_vgg_loss = perceptual(predict_vgg_feature, gt_vgg_feature)
        
        # GAN loss
        output = predict_imgs * smooth_masks + imgs * (1.- smooth_masks)
        #adv_loss = -discriminator(output).mean()
        adv_loss = g_nonsaturating_loss(discriminator(output))

        total_loss = 1.0 * predict_l1_loss + 1.0 * predict_noise_loss\
            + 0.1 * predict_vgg_loss + 0.001 * adv_loss

        loss_dict["predict_l1"] = predict_l1_loss
        loss_dict["predict_vgg"] = predict_vgg_loss
        loss_dict["predict_noise"] = predict_noise_loss
        loss_dict["adv_g"] =  adv_loss
        loss_dict["real_d"] = real_d_loss
        loss_dict["fake_d"] = fake_d_loss
        loss_dict["d_loss"] = d_loss


        gatedconv_optim.zero_grad()
        total_loss.backward()
        gatedconv_optim.step()
        
        loss_reduced = reduce_loss_dict(loss_dict)

        predict_l1_val = loss_reduced["predict_l1"].mean().item()
        predict_vgg_val = loss_reduced["predict_vgg"].mean().item()
        predict_noise_val = loss_reduced["predict_noise"].mean().item()
        adv_g_val = loss_reduced["adv_g"].mean().item()
        real_d_val = loss_reduced["real_d"].mean().item()
        fake_d_val = loss_reduced["fake_d"].mean().item()
        d_loss_val = loss_reduced["d_loss"].mean().item()

        if get_rank() == 0:
            if i % 1000 == 0:
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
                utils.save_image(
                        blend_imgs,
                        f"{str(output_dir)}/noise.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                utils.save_image(
                        predict_noise,
                        f"{str(output_dir)}/predict_noise.png",
                        nrow=int(4),
                        normalize=True,
                        value_range=(-1, 1),
                    )
            pbar.set_description(
                (
                    f"l1: {predict_l1_val:.2f};"
                    f"vgg: {predict_vgg_val:.2f};"
                    f"ns: {predict_noise_val:.2f};"
                    f"real_d: {real_d_val:.2f};"
                    f"fake_d: {fake_d_val:.2f};"
                    f"d_ls: {d_loss_val:.2f};"
                    f"adv_g: {adv_g_val:.2f};"
                )
            )
            if wandb and args.wandb:
                wandb.log(
                    {
                        "Predict_l1": predict_l1_val,
                        "Predict_Noise": predict_noise_val,
                        "Predict_VGG": predict_vgg_val,
                        "Predict_adv": adv_g_val,
                        "real_d": real_d_val,
                        "fake_d": fake_d_val,
                        "d_loss": d_loss_val
                    }
                )

            if i % 5000 == 0:
                torch.save(
                    {
                        "discriminator": discriminator_module.state_dict(),
                        "gatedconv": gatedconv_module.state_dict(),
                        "args": args
                    },
                    f"{str(output_dir)}/checkpoint.pt",
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
        "--iter", type=int, default=550001, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
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
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--threshold", type=float, help="learning rate")
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--use_gtmask", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    args = parser.parse_args()

    # n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = n_gpu > 1

    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
    #     synchronize()

    args.start_iter = 0

    gatedconv = None
    gatedconv_optim = None
    val_list = []

    from models.CPN_IIN import RR_inpaint_model_minus_trans
    from mpn_model_rename.architecture import Discriminator

    if args.use_gtmask is True:
        gatedconv = RR_inpaint_model_minus_trans().to(device)

    discriminator = Discriminator().to(device)

    gatedconv_optim = optim.Adam(
        gatedconv.parameters(),
        lr=args.lr
    )
    
    discriminator_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr
    )
    

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass
        

    # if args.distributed:
    #     gatedconv = nn.parallel.DistributedDataParallel(
    #         gatedconv,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         broadcast_buffers=False,
    #     )


    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)


    from dataset_train import Places2_strokemask, FFHQ_strokemask, ImageNet_strokemask

    if args.dataset == "ffhq":
        main_dataset = FFHQ_strokemask()
    elif args.dataset == "places2":
        main_dataset = Places2_strokemask()
    elif args.dataset == "imagenet":
        main_dataset = ImageNet_strokemask()
    
    main_loader = data.DataLoader(
        main_dataset,
        batch_size=args.batch,
        sampler=data_sampler(main_dataset, shuffle=True, distributed=False),
        num_workers=10,
        drop_last=True,
    )

    if get_rank() == 0:
        if wandb is not None and args.wandb:
            wandb.init(project="PLUS GAN test")
        

    train(args, main_loader,gatedconv, discriminator, gatedconv_optim, discriminator_optim, device)
