import torch
import torch.nn as nn
import math
from gated_model.networks import Conv2dWithActivation, DeConv2dWithActivation, get_pad
from gated_model.networks import GatedConv2dWithActivation, GatedDeConv2dWithActivation
from .Stripformer import Inter_SA, Intra_SA


# Contaminant prediction network
class Noisenet(torch.nn.Module):
    def __init__(self, n_in_channel=4):
        super(Noisenet, self).__init__()
        cnum = 32
        # just preserve coarse net since model size
        self.e_res_1 = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
        )

        self.e_res_2 = nn.Sequential(
            # downsample 128
            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )

        self.e_res_3 = nn.Sequential(
            #downsample to 64
            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )

        self.atrous = nn.Sequential(
            # atrous convlution
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )

        self.d_res_2 = nn.Sequential(
            # upsample
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )

        self.d_res_1 = nn.Sequential(
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            
        )

        self.proj = nn.Sequential(
            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )


    def forward(self, input_imgs):
        # Coarse
        x_e_1 = self.e_res_1(input_imgs)
        x_e_2 = self.e_res_2(x_e_1)
        x_e_3 = self.e_res_3(x_e_2)
        
        x_d_3 = self.atrous(x_e_3)
        x_d_2 = self.d_res_2(x_e_3 + x_d_3)
        x_d_1 = self.d_res_1(x_e_2 + x_d_2)

        x = self.proj(x_e_1 + x_d_1)
        x = torch.clamp(x, -1., 1.)

        return x , x_e_1 + x_d_1
    

# Image Inpainting network    
class Inpaintnet_trans(torch.nn.Module):
    
    def __init__(self, n_in_channel=4):
        super(Inpaintnet_trans, self).__init__()
        head_num = 5
        dim = 320
        cnum = 32
        # just preserve coarse net since model size
        self.e_res_1 = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
        )

        self.e_res_2 = nn.Sequential(
            # downsample 128
            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )

        self.e_res_3 = nn.Sequential(
            #downsample to 64
            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )

        self.e_res_4 = nn.Sequential(
            #downsample to 32
            GatedConv2dWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 4, 2)),
            GatedConv2dWithActivation(8*cnum, dim, 3, 1, padding=get_pad(32, 3, 1)),
        )

        self.transblock = nn.Sequential(
            Intra_SA(dim, head_num),
            Inter_SA(dim, head_num),
            Intra_SA(dim, head_num),
            Inter_SA(dim, head_num),
            Intra_SA(dim, head_num),
            Inter_SA(dim, head_num),
            Intra_SA(dim, head_num),
            Inter_SA(dim, head_num),
            Intra_SA(dim, head_num),
            Inter_SA(dim, head_num),
        )

        self.d_res_3 = nn.Sequential(
            # upsample
            GatedDeConv2dWithActivation(2, dim, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )

        self.d_res_2 = nn.Sequential(
            # upsample
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )

        self.d_res_1 = nn.Sequential(
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            
        )

        self.proj = nn.Sequential(
            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)
        )


    def forward(self, input_imgs, noise_feature):
        # Coarse
        
        x_e_1 = self.e_res_1(input_imgs)
        x_e_1 = x_e_1 + noise_feature

        x_e_2 = self.e_res_2(x_e_1)
        x_e_3 = self.e_res_3(x_e_2)
        x_e_4 = self.e_res_4(x_e_3)
        
        x_d_4 = self.transblock(x_e_4)
        x_d_3 = self.d_res_3(x_e_4 + x_d_4)
        x_d_2 = self.d_res_2(x_e_3 + x_d_3)
        x_d_1 = self.d_res_1(x_e_2 + x_d_2)

        x = self.proj(x_e_1 + x_d_1)
        x = torch.clamp(x, -1., 1.)

        return x
    

# CPN & IIN
class RR_inpaint_model_minus_trans(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self):
        super(RR_inpaint_model_minus_trans, self).__init__()
        cnum = 32

        self.inpaint_net = Inpaintnet_trans()
        self.noise_net = Noisenet()

        self.mask_proj = nn.Sequential(
            Conv2dWithActivation(1, cnum//2, 1, 1),
            Conv2dWithActivation(cnum//2, cnum, 1, 1, activation=None),
            nn.Sigmoid()
        )

        # change to conv2d
        self.noise_proj = nn.Sequential(
            Conv2dWithActivation(cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            #Conv2dWithActivation(cnum, cnum, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

    def forward(self, masked_imgs, masks):
        input_imgs = torch.cat([masked_imgs, masks], dim = 1)
        # Noise net
        noise, noise_feature = self.noise_net(input_imgs)

        # mask projeciton
        mask_proj_feature = self.mask_proj(masks)

        # Noise feature projection
        noise_proj_feature = self.noise_proj(noise_feature)
        noise_proj_feature = noise_proj_feature * mask_proj_feature * -1.

        # Input inpaintnet
        inpaint_image = self.inpaint_net(input_imgs, noise_proj_feature)

        return noise, inpaint_image