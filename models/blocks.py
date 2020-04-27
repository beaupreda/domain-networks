"""
gathering of blocks/group of layers used in domainnet.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        """
        represents the operations of a fully connected layer (require parameters) and ReLU (no parameters).
        :param in_dim: number of channels for the input.
        :param out_dim: number of channels for the output.
        :param bias: learn the linear bias or not.
        """
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass implementation (relu -> fc)
        :param x: input tensor.
        :return: tensor.
        """
        return F.relu(self.linear(x))


class Conv2dBN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ksize: (int, int), stride: int = 1,
                 padding: int = 0, dilation: int = 1, bias: bool = True):
        """
        represents the operations of 2d convolution and batch normalization (require parameters).
        :param in_dim: number of channels for the input.
        :param out_dim: number of channels for the output.
        :param ksize: size of the convolution kernel.
        :param stride: distance between consecutive convolutions.
        :param padding: number of pixels added on the contour of the tensor.
        :param dilation: distance between pixels considered by the convolutions kernel.
        :param bias: learn bias of convolution or not.
        """
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass implementation (batch normalization -> convolution).
        :param x: input tensor.
        :return: tensor.
        """
        return self.bn(self.conv(x))


class Conv2dBNReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ksize: (int, int), stride: int = 1,
                 padding: int = 0, dilation: int = 1, bias: bool = True):
        """
        represents the operations of 2d convolution, batch normalization (require parameters) and ReLU (no parameters).
        :param in_dim: number of channels for the input.
        :param out_dim: number of channels for the output.
        :param ksize: size of the convolution kernel.
        :param stride: distance between consecutive convolutions.
        :param padding: number pixels added on the contour of the tensor.
        :param dilation: distance between pixels considered by the convolution kernel.
        :param bias: learn bias of convolution or not.
        """
        super(Conv2dBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass implementation (relu -> batch normalization -> convolution).
        :param x: input tensor.
        :return: tensor.
        """
        return F.relu(self.bn(self.conv(x)))
