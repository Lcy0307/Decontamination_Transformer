import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from mpn_model_rename.base import BaseNetwork
from mpn_model_rename.blocks_bn import ResBlock, ConvBlock

from torch.autograd import Variable
import torch.autograd as autograd

import numpy as np

# https://github.com/birdortyedi/vcnet-blind-image-inpainting
class MPN(BaseNetwork):
    def __init__(self, base_n_channels = 64, neck_n_channels = 128):
        super(MPN, self).__init__()
        assert base_n_channels >= 4, "Base num channels should be at least 4"
        assert neck_n_channels >= 16, "Neck num channels should be at least 16"
        self.rb1 = ResBlock(channels_in=3, channels_out=base_n_channels, kernel_size=5, stride=2, padding=2, dilation=1)
        self.rb2 = ResBlock(channels_in=base_n_channels, channels_out=base_n_channels * 2, kernel_size=3, stride=2)
        self.rb3 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.rb4 = ResBlock(channels_in=base_n_channels * 2, channels_out=neck_n_channels, kernel_size=3, stride=1, padding=4, dilation=4)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.rb5 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1)
        self.rb6 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels, kernel_size=3, stride=1)
        self.rb7 = ResBlock(channels_in=base_n_channels, channels_out=base_n_channels // 2, kernel_size=3, stride=1)

        self.cb1 = ConvBlock(channels_in=base_n_channels // 2, channels_out=base_n_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(base_n_channels // 4, 1, kernel_size=3, stride=1, padding=1)

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, x):
        out = self.rb1(x)
        out = self.rb2(out)
        out = self.rb3(out)
        neck = self.rb4(out)
        # bottleneck here

        out = self.rb5(neck)
        out = self.upsample(out)
        out = self.rb6(out)
        out = self.upsample(out)
        out = self.rb7(out)

        out = self.cb1(out)
        out = self.conv1(out)

        return torch.sigmoid(out)

    
class Discriminator(BaseNetwork):
    def __init__(self, base_n_channels=64):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.image_to_features = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(4 * base_n_channels, 8 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * base_n_channels * 8 * 8
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1)
        )

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)