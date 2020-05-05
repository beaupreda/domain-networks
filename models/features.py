"""
architecture of the domain feature extractors.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import torch
import torch.nn as nn

import models.blocks as blk


class Features(nn.Module):
    def __init__(self, num_channels: int):
        """
        represents the feature extractors for each spectrum.
        :param num_channels: number of channels of the input image.
        """
        super(Features, self).__init__()
        self.conv1 = blk.Conv2dBNReLU(in_dim=num_channels, out_dim=32, ksize=(5, 5))
        self.conv2 = blk.Conv2dBNReLU(in_dim=32, out_dim=64, ksize=(5, 5))
        self.conv3 = blk.Conv2dBNReLU(in_dim=64, out_dim=64, ksize=(5, 5))
        self.conv4 = blk.Conv2dBNReLU(in_dim=64, out_dim=64, ksize=(5, 5))
        self.conv5 = blk.Conv2dBNReLU(in_dim=64, out_dim=128, ksize=(5, 5))
        self.conv6 = blk.Conv2dBNReLU(in_dim=128, out_dim=128, ksize=(5, 5))
        self.conv7 = blk.Conv2dBNReLU(in_dim=128, out_dim=256, ksize=(5, 5))
        self.conv8 = blk.Conv2dBNReLU(in_dim=256, out_dim=256, ksize=(5, 5))
        self.conv9 = blk.Conv2dBN(in_dim=256, out_dim=256, ksize=(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass implementation.
        :param x: input tensor.
        :return: tensor.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)
        y = self.conv9(y)
        return y
