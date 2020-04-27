"""
architecture of the classifier heads.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.blocks as blk


class Classifier(nn.Module):
    def __init__(self, num_channels: int):
        """
        represents the correlation and concatenation classifying heads.
        :param num_channels: feature dimension of the merged vector.
        """
        super(Classifier, self).__init__()
        self.fc1 = blk.LinearReLU(in_dim=num_channels, out_dim=128)
        self.fc2 = blk.LinearReLU(in_dim=128, out_dim=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass implementation.
        :param x: input tensor.
        :return: 2 elements probability tensor.
        """
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y
