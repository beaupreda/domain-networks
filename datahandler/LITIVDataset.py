"""
implementation of the train and test LITIV datasets.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

from argparse import Namespace
from typing import List

import cv2
import numpy as np
import torch

import utils.io as io
import utils.misc as misc


class TrainLITIVDataset:
    def __init__(self, rgb: List[str], lwir: List[str], disparity: str, phase: str, args: Namespace):
        """
        represents the LITIV dataset used for training and validation.
        :param rgb: list of paths to rgb images.
        :param lwir: list of paths to lwir images.
        :param disparity: file contraining all data points.
        :param args: structure containing all arguments.
        """
        self.rgb = rgb
        self.lwir = lwir
        self.bsize = args.batch_size
        self.psize = args.patch_size
        self.channels = 3
        self.ptr = 0
        tmp_disparity = io.read_disparity_gt(disparity)

        # positive match offset = {-1, 0, 1}
        positive_offset = np.arange(start=-1, stop=2, dtype=np.int32)
        # negative match offset = {-30, -29, ..., -11, -10} and {10, 11, ..., 29, 30}
        negative_offset = np.hstack((np.arange(start=-30, stop=-9, dtype=np.int32),
                                     np.arange(start=10, stop=31, dtype=np.int32)))

        self.disparity = misc.create_positive_negative_samples(tmp_disparity, positive_offset, negative_offset)

        print(f'total {phase} locations: {self.disparity.shape[0]}')

        patch_size = 2 * self.psize
        self.patch_rgb = torch.zeros(size=(self.bsize, self.channels, patch_size, patch_size), dtype=torch.float32)
        self.patch_lwir = torch.zeros(size=(self.bsize, self.channels, patch_size, patch_size), dtype=torch.float32)
        self.targets = torch.zeros(size=(self.bsize, ), dtype=torch.int64)

    def get_batch(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        creates patches from the dataset.
        :return: batched tensors of patches and the targets.
        """
        for idx in range(self.bsize):
            i = self.ptr + idx
            if i > self.disparity.shape[0] - 1:
                i = 0
                self.ptr = -idx
            frame, x, y, dx, label = self.disparity[i]

            rgb = cv2.imread(self.rgb[frame], cv2.IMREAD_COLOR).astype(np.float32)
            lwir = cv2.imread(self.lwir[frame], cv2.IMREAD_COLOR).astype(np.float32)

            rgb = torch.from_numpy(misc.preprocess(rgb, True))
            lwir = torch.from_numpy(misc.preprocess(lwir, True))

            self.patch_rgb[idx] = rgb[:, y - self.psize:y + self.psize,
                                      x - self.psize:x + self.psize]
            self.patch_lwir[idx] = lwir[:, y - self.psize:y + self.psize,
                                        dx - self.psize:dx + self.psize]
            self.targets[idx] = torch.tensor(data=label, dtype=torch.int64)

        self.ptr += self.bsize
        return self.patch_rgb, self.patch_lwir, self.targets


class TestLITIVDataset:
    def __init__(self, rgb: List[str], lwir: List[str], disparity: str, args: Namespace):
        """
        represents the LITIV dataset used for testing.
        :param rgb: list of paths to rgb images.
        :param lwir: list of paths to lwir images.
        :param disparity: file contraining all data points.
        :param args: structure containing all user arguments.
        """
        self.rgb = rgb
        self.lwir = lwir
        self.bsize = args.batch_size
        self.psize = args.patch_size
        self.hdisp = int(float(args.max_disparity) / 2.0)
        self.channels = 3
        self.ptr = 0
        self.disparity = io.read_disparity_gt(disparity)

        print(f'total testing locations: {self.disparity.shape[0]}')

        patch_size = 2 * self.psize
        range_size = 2 * self.hdisp + patch_size
        self.patch_rgb = torch.zeros(size=(self.bsize, self.channels, patch_size, patch_size), dtype=torch.float32)
        self.patch_lwir = torch.zeros(size=(self.bsize, self.channels, patch_size, range_size), dtype=torch.float32)
        self.targets = torch.zeros(size=(self.bsize, ), dtype=torch.int64)

        self.remainder = self.disparity.shape[0] % self.bsize
        self.last_batch_idx = self.disparity.shape[0] - self.remainder

    def get_batch(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        creates testing patches from the dataset.
        :return: batched tensor of patches and the targets.
        """
        for idx in range(self.bsize):
            i = self.ptr + idx
            # last batch is smaller
            if i == self.last_batch_idx:
                patch_size = 2 * self.psize
                range_size = 2 * self.hdisp + patch_size
                self.patch_rgb = torch.zeros(size=(self.bsize, self.channels, patch_size, patch_size),
                                             dtype=torch.float32)
                self.patch_lwir = torch.zeros(size=(self.bsize, self.channels, patch_size, range_size),
                                              dtype=torch.float32)
                self.targets = torch.zeros(size=(self.bsize,), dtype=torch.int64)

            frame, x, y, dx = self.disparity[i]

            rgb = cv2.imread(self.rgb[frame], cv2.IMREAD_COLOR).astype(np.float32)
            lwir = cv2.imread(self.lwir[frame], cv2.IMREAD_COLOR).astype(np.float32)

            rgb = torch.from_numpy(misc.preprocess(rgb, True))
            lwir = torch.from_numpy(misc.preprocess(lwir, True))

            self.patch_rgb[idx] = rgb[:, y - self.psize:y + self.psize,
                                      x - self.psize:x + self.psize]
            self.patch_lwir[idx] = lwir[:, y - self.psize:y + self.psize,
                                        dx - self.psize - self.hdisp:dx + self.psize + self.hdisp]
            self.targets[idx] = torch.tensor(data=self.hdisp, dtype=torch.int64)

            if i >= self.last_batch_idx and idx == self.remainder - 1:
                return self.patch_rgb, self.patch_lwir, self.targets

        self.ptr += self.bsize
        return self.patch_rgb, self.patch_lwir, self.targets
