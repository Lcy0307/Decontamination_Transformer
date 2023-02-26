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


def train(args, main_loader, mask_model, mask_model_optim, device):
    main_loader = sample_data(main_loader)
    
    smoother = ConfidenceDrivenMaskLayer_input_iter().to(device)
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}

    if args.distributed:
        mask_model_module = mask_model

    else:
        mask_model_module = mask_model


    mask_model.train()

    for idx in pbar:
        epoch = 0
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        gt, noise, masks= next(main_loader)
        
        noise = noise.to(device)
        masks = masks.to(device)
        gt = gt.to(device)
        

        with torch.no_grad():
            # Quantization
            masks[masks > 0.5] = 1.
            masks[masks < 0.5] = 0.
            
            for j in range(gt.shape[0]):
                masks[j:j+1] = masks[j:j+1] + smoother(1. - masks[j:j+1])
            masks = torch.clamp(masks, min=0., max=1.)
            corrupt_imgs = noise * masks + gt * (1. - masks)

        predict_masks = mask_model(corrupt_imgs.detach())
        
        criterion_mask = nn.MSELoss()
        predict_mask_loss = criterion_mask(predict_masks,masks.detach())

        total_loss = predict_mask_loss

        loss_dict["predict_mask_loss"] = predict_mask_loss


        mask_model_optim.zero_grad()
        total_loss.backward()

        mask_model_optim.step()
        loss_reduced = reduce_loss_dict(loss_dict)

        predict_mask_loss_val = loss_reduced["predict_mask_loss"].mean().item()

        if get_rank() == 0:
            # if i % 1000 == 0:
            #     mask_model.eval()
            #     with torch.no_grad():
            #         val_gt, val_noise, val_mask_smooths, val_masks = val_list
            #         val_smooth_img = val_noise * val_mask_smooths + val_gt * (1. - val_mask_smooths)
            #         val_hard_img = val_noise * val_masks + val_gt * (1. - val_masks)
            #         # smooth mask prediction
            #         predict_mask_smooth = mask_model(val_smooth_img)
            #         mask_smooth_loss = criterion_mask(predict_mask_smooth,val_mask_smooths)
            #         #sharp mask prediction
            #         predict_mask_hard = mask_model(val_hard_img)
            #         mask_sharp_loss = criterion_mask(predict_mask_hard,val_masks)
            #     mask_model.train()
            pbar.set_description(
                (
                    f"mask_loss: {predict_mask_loss_val:.4f};"
                    # f"mse_smooth: {mask_smooth_loss:.4f};"
                    # f"mse_sharp: {mask_sharp_loss:.4f};"
                )
            )
            if wandb and args.wandb:
                wandb.log(
                    {
                        "mask_l2": predict_mask_loss,
                        # "mse_smooth_mask": mask_smooth_loss,
                        # "mse_sharp_mask": mask_sharp_loss,
                    }
                )
            if i % 5000 == 0:
                torch.save(
                    {
                        "mask_model": mask_model_module.state_dict(),
                        "args": args
                    },
                    f"{str(output_dir)}/checkpoint.pt",
                )
                utils.save_image(
                        masks,
                        f"{str(output_dir)}/masks.png",
                        nrow=int(4),
                        normalize=True,
                        range=(0, 1),
                    )
                utils.save_image(
                        predict_masks,
                        f"{str(output_dir)}/predict_masks.png",
                        nrow=int(4),
                        normalize=True,
                        range=(0, 1),
                    )
                utils.save_image(
                        corrupt_imgs,
                        f"{str(output_dir)}/corrupt_img.png",
                        nrow=int(4),
                        normalize=True,
                        range=(-1, 1),
                    )
                utils.save_image(
                        noise,
                        f"{str(output_dir)}/noise.png",
                        nrow=int(4),
                        normalize=True,
                        range=(-1, 1),
                    )
                utils.save_image(
                        gt,
                        f"{str(output_dir)}/imgs.png",
                        nrow=int(4),
                        normalize=True,
                        range=(-1, 1),
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
        default="ffhq",
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

    from mpn_model_rename.architecture import MPN

    mask_model = MPN().to(device)

    mask_model_optim = optim.Adam(
        mask_model.parameters(),
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
        

    if args.distributed:
        mask_model = nn.parallel.DistributedDataParallel(
            mask_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )


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
        sampler=data_sampler(main_dataset, shuffle=True, distributed=args.distributed),
        num_workers=10,
        drop_last=True,
    )

    if get_rank() == 0:
        if wandb is not None and args.wandb:
            wandb.init(project="MPN final test")
        
        

    train(args, main_loader, mask_model, mask_model_optim, device)
