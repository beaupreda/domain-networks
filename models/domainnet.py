"""
proposed model with joint training of correlation and concatenation branches, with different feature extractions.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.features import Features
from models.classifier import Classifier


class DomainNet(nn.Module):
    def __init__(self, num_channels: int):
        """
        represents the architecture of the proposed model.
        :param num_channels: number of channels of the input image.
        """
        super(DomainNet, self).__init__()
        self.rgb_features = Features(num_channels=num_channels)
        self.lwir_features = Features(num_channels=num_channels)
        self.corr_cls = Classifier(num_channels=256)
        self.concat_cls = Classifier(num_channels=512)

    def forward(self, rgb: torch.Tensor, lwir: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        forward pass implementation of both correlation and concatenation branches.
        :param rgb: rgb patch tensor.
        :param lwir: lwir patch tensor.
        :return: 2 elements probability tensors (rgb and lwir being the same or not).
        """
        rgb = self.rgb_features(rgb)
        lwir = self.lwir_features(lwir)

        correlation = torch.matmul(rgb, lwir)
        concatenation = torch.cat((F.relu(rgb), F.relu(lwir)), dim=1)

        correlation = correlation.view(correlation.size(0), -1)
        concatenation = concatenation.view(concatenation.size(0), -1)

        correlation = self.corr_cls(correlation)
        concatenation = self.concat_cls(concatenation)

        return correlation, concatenation
