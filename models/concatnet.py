"""
proposed model with only the concatenation branch.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.features import Features
from models.classifier import Classifier


class ConcatNet(nn.Module):
    def __init__(self, num_channels: int):
        """
        represents the architecture of the proposed model with only the concatenation branch.
        :param num_channels: number of channels of the input image.
        """
        super(ConcatNet, self).__init__()
        self.rgb_features = Features(num_channels=num_channels)
        self.lwir_features = Features(num_channels=num_channels)
        self.concat_cls = Classifier(num_channels=512)

    def forward(self, rgb: torch.Tensor, lwir: torch.Tensor) -> torch.Tensor:
        """
        forward pass implementation of the concatenation branch.
        :param rgb: rgb patch tensor.
        :param lwir: lwir patch tensor.
        :return: 2 elements probability tensor (rgb and lwir being the same or not).
        """
        rgb = self.rgb_features(rgb)
        lwir = self.lwir_features(lwir)

        concatenation = torch.cat((F.relu(rgb), F.relu(lwir)), dim=1)
        concatenation = concatenation.view(concatenation.size(0), -1)
        concatenation = self.concat_cls(concatenation)

        return concatenation
