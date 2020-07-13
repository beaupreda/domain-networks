"""
combination of LITIV 2014 and 2018 datasets.

author: David-Alexandre Beaupre
date: 2020-04-24
"""

import glob
import os
import random
from collections import defaultdict
from typing import DefaultDict, List

import utils.io as io
from datahandler.LITIV2014 import LITIV2014
from datahandler.LITIV2018 import LITIV2018


class LITIV:
    def __init__(self, root: str = None, psize: int = None, fold: int = None):
        """
        represents the whole LITIV dataset with training/validation/testing data split.
        :param root: path to the folder containing both LITIV 2014 and LITIV 2018 folder.
        :param fold: number identifying which fold to keep as testing data.
        :param psize: half size of the patch.
        """
        random.seed(42)
        self.root = root
        self.num_val = 30
        self.litiv2018 = LITIV2018(root, psize, fold=fold)
        self.litiv2014 = LITIV2014(root, psize, fold=fold)
        self.rgb = defaultdict(list)
        self.lwir = defaultdict(list)
        self.rmask = defaultdict(list)
        self.lmask = defaultdict(list)
        self.disp = {}

        # TODO: change that
        if fold > 3:
            self.num_val = 150

        self._prepare()
        self._split()

    @staticmethod
    def _make_gt(root: str, disparity: List[str]) -> str:
        """
        creates gt.txt file for each split.
        :param root: location to create the file.
        :param disparity: list of disparity files for a given split.
        :return: path to gt.txt
        """
        print(f'creating ground-truth files...')
        data_points = []
        for i, d in enumerate(disparity):
            points = io.read_disparity(d, i)
            data_points.extend(points)
        random.shuffle(data_points)
        gt = os.path.join(root, 'gt.txt')
        with open(gt, 'w') as file:
            for i, x, y, dx in data_points:
                file.write(f'{i} {x} {y} {dx}\n')
        return gt

    def _make_images(self, root: str, data: DefaultDict[str, List[str]], split: str) -> List[str]:
        """
        creates rgb/lwir images folders and copy images for each split.
        :param root: locations of the rgb/lwir folders.
        :param data: list of images.
        :param split: train/validation/test.
        :return: list of disparity file for this split.
        """
        print(f'creating images...')
        disparity = []
        for i, (r, l, mr, ml, d) in enumerate(zip(data['rgb'], data['lwir'], data['mask_rgb'], data['mask_lwir'],
                                                  data['disparity'])):
            rgb_name = os.path.join(root, split, 'rgb', f'{i}.png')
            lwir_name = os.path.join(root, split, 'lwir', f'{i}.png')
            mask_rgb_name = os.path.join(root, split, 'mask_rgb', f'{i}.png')
            mask_lwir_name = os.path.join(root, split, 'mask_lwir', f'{i}.png')
            io.copy_image(r, rgb_name)
            io.copy_image(l, lwir_name)
            io.copy_image(mr, mask_rgb_name)
            io.copy_image(ml, mask_lwir_name)
            self.rgb[split].append(rgb_name)
            self.lwir[split].append(lwir_name)
            self.rmask[split].append(mask_rgb_name)
            self.lmask[split].append(mask_lwir_name)
            disparity.append(d)
        return disparity

    def _prepare(self) -> None:
        """
        creates relevant folder for the LITIV dataset.
        :return: void.
        """
        dataset_root = os.path.join(self.root, 'dataset')
        if not os.path.isdir(dataset_root):
            os.mkdir(dataset_root)

        splits = ['train', 'validation', 'test']
        spectrums = ['rgb', 'lwir', 'mask_rgb', 'mask_lwir']
        for split in splits:
            split_root = os.path.join(dataset_root, split)
            if not os.path.isdir(split_root):
                os.mkdir(split_root)
            for spectrum in spectrums:
                spectrum_root = os.path.join(split_root, spectrum)
                if not os.path.isdir(spectrum_root):
                    os.mkdir(spectrum_root)
                # remove all files, they will be recreated
                files = glob.glob(os.path.join(spectrum_root, '*'))
                for file in files:
                    os.remove(file)

    def _split(self) -> None:
        """
        splits the LITIV dataset into train/validation/test
        :return: void.
        """
        print(f'splitting dataset...')
        train_data = defaultdict(list)
        val_data = defaultdict(list)
        test_data = defaultdict(list)
        if self.litiv2014.fold == 1 or self.litiv2014.fold == 2 or self.litiv2014.fold == 3:
            for video in self.litiv2018.rgb.keys():
                train_data['rgb'].extend(self.litiv2018.rgb[video])
                train_data['lwir'].extend(self.litiv2018.lwir[video])
                train_data['mask_rgb'].extend(self.litiv2018.mask_rgb[video])
                train_data['mask_lwir'].extend(self.litiv2018.mask_lwir[video])
                train_data['disparity'].extend(self.litiv2018.disparity[video])

            if self.litiv2014.fold == 1:
                test_data['rgb'].extend(self.litiv2014.rgb['vid1'])
                test_data['lwir'].extend(self.litiv2014.lwir['vid1'])
                test_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid1'])
                test_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid1'])
                test_data['disparity'].extend(self.litiv2014.disparity['vid1'])
                val_data['rgb'].extend(self.litiv2014.rgb['vid2'])
                val_data['lwir'].extend(self.litiv2014.lwir['vid2'])
                val_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid2'])
                val_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid2'])
                val_data['disparity'].extend(self.litiv2014.disparity['vid2'])
                val_data['rgb'].extend(self.litiv2014.rgb['vid3'])
                val_data['lwir'].extend(self.litiv2014.lwir['vid3'])
                val_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid3'])
                val_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid3'])
                val_data['disparity'].extend(self.litiv2014.disparity['vid3'])
            elif self.litiv2014.fold == 2:
                test_data['rgb'].extend(self.litiv2014.rgb['vid2'])
                test_data['lwir'].extend(self.litiv2014.lwir['vid2'])
                test_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid2'])
                test_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid2'])
                test_data['disparity'].extend(self.litiv2014.disparity['vid2'])
                val_data['rgb'].extend(self.litiv2014.rgb['vid1'])
                val_data['lwir'].extend(self.litiv2014.lwir['vid1'])
                val_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid1'])
                val_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid1'])
                val_data['disparity'].extend(self.litiv2014.disparity['vid1'])
                val_data['rgb'].extend(self.litiv2014.rgb['vid3'])
                val_data['lwir'].extend(self.litiv2014.lwir['vid3'])
                val_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid3'])
                val_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid3'])
                val_data['disparity'].extend(self.litiv2014.disparity['vid3'])
            elif self.litiv2014.fold == 3:
                test_data['rgb'].extend(self.litiv2014.rgb['vid3'])
                test_data['lwir'].extend(self.litiv2014.lwir['vid3'])
                test_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid3'])
                test_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid3'])
                test_data['disparity'].extend(self.litiv2014.disparity['vid3'])
                val_data['rgb'].extend(self.litiv2014.rgb['vid1'])
                val_data['lwir'].extend(self.litiv2014.lwir['vid1'])
                val_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid1'])
                val_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid1'])
                val_data['disparity'].extend(self.litiv2014.disparity['vid1'])
                val_data['rgb'].extend(self.litiv2014.rgb['vid2'])
                val_data['lwir'].extend(self.litiv2014.lwir['vid2'])
                val_data['mask_rgb'].extend(self.litiv2014.mask_rgb['vid2'])
                val_data['mask_lwir'].extend(self.litiv2014.mask_lwir['vid2'])
                val_data['disparity'].extend(self.litiv2014.disparity['vid2'])
        elif self.litiv2018.fold == 4 or self.litiv2018.fold == 5 or self.litiv2018.fold == 6:
            for video in self.litiv2014.rgb.keys():
                train_data['rgb'].extend(self.litiv2014.rgb[video])
                train_data['lwir'].extend(self.litiv2014.lwir[video])
                train_data['mask_rgb'].extend(self.litiv2014.mask_rgb[video])
                train_data['mask_lwir'].extend(self.litiv2014.mask_lwir[video])
                train_data['disparity'].extend(self.litiv2014.disparity[video])

            if self.litiv2018.fold == 4:
                test_data['rgb'].extend(self.litiv2018.rgb['vid04'])
                test_data['lwir'].extend(self.litiv2018.lwir['vid04'])
                test_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid04'])
                test_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid04'])
                test_data['disparity'].extend(self.litiv2018.disparity['vid04'])
                val_data['rgb'].extend(self.litiv2018.rgb['vid07'])
                val_data['lwir'].extend(self.litiv2018.lwir['vid07'])
                val_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid07'])
                val_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid07'])
                val_data['disparity'].extend(self.litiv2018.disparity['vid07'])
                val_data['rgb'].extend(self.litiv2018.rgb['vid08'])
                val_data['lwir'].extend(self.litiv2018.lwir['vid08'])
                val_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid08'])
                val_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid08'])
                val_data['disparity'].extend(self.litiv2018.disparity['vid08'])
            elif self.litiv2018.fold == 5:
                test_data['rgb'].extend(self.litiv2018.rgb['vid07'])
                test_data['lwir'].extend(self.litiv2018.lwir['vid07'])
                test_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid07'])
                test_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid07'])
                test_data['disparity'].extend(self.litiv2018.disparity['vid07'])
                val_data['rgb'].extend(self.litiv2018.rgb['vid04'])
                val_data['lwir'].extend(self.litiv2018.lwir['vid04'])
                val_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid04'])
                val_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid04'])
                val_data['disparity'].extend(self.litiv2018.disparity['vid04'])
                val_data['rgb'].extend(self.litiv2018.rgb['vid08'])
                val_data['lwir'].extend(self.litiv2018.lwir['vid08'])
                val_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid08'])
                val_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid08'])
                val_data['disparity'].extend(self.litiv2018.disparity['vid08'])
            elif self.litiv2018.fold == 6:
                test_data['rgb'].extend(self.litiv2018.rgb['vid08'])
                test_data['lwir'].extend(self.litiv2018.lwir['vid08'])
                test_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid08'])
                test_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid08'])
                test_data['disparity'].extend(self.litiv2018.disparity['vid08'])
                val_data['rgb'].extend(self.litiv2018.rgb['vid04'])
                val_data['lwir'].extend(self.litiv2018.lwir['vid04'])
                val_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid04'])
                val_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid04'])
                val_data['disparity'].extend(self.litiv2018.disparity['vid04'])
                val_data['rgb'].extend(self.litiv2018.rgb['vid07'])
                val_data['lwir'].extend(self.litiv2018.lwir['vid07'])
                val_data['mask_rgb'].extend(self.litiv2018.mask_rgb['vid07'])
                val_data['mask_lwir'].extend(self.litiv2018.mask_lwir['vid07'])
                val_data['disparity'].extend(self.litiv2018.disparity['vid07'])

        # shuffle validation data the same way for rgb/lwir/disparity.
        data = list(zip(val_data['rgb'], val_data['lwir'], val_data['mask_rgb'],
                        val_data['mask_lwir'], val_data['disparity']))
        random.shuffle(data)
        val_data['rgb'], val_data['lwir'], val_data['mask_rgb'], val_data['mask_lwir'], val_data['disparity'] = \
            zip(*data)
        val_data['rgb'] = list(val_data['rgb'])
        val_data['lwir'] = list(val_data['lwir'])
        val_data['mask_rgb'] = list(val_data['mask_rgb'])
        val_data['mask_lwir'] = list(val_data['mask_lwir'])
        val_data['disparity'] = list(val_data['disparity'])

        # only keep self.num_val images for validation, rest go in training.
        train_data['rgb'].extend(val_data['rgb'][self.num_val:])
        train_data['lwir'].extend(val_data['lwir'][self.num_val:])
        train_data['mask_rgb'].extend(val_data['mask_rgb'][self.num_val:])
        train_data['mask_lwir'].extend(val_data['mask_lwir'][self.num_val:])
        train_data['disparity'].extend(val_data['disparity'][self.num_val:])
        del val_data['rgb'][self.num_val:]
        del val_data['lwir'][self.num_val:]
        del val_data['mask_rgb'][self.num_val:]
        del val_data['mask_lwir'][self.num_val:]
        del val_data['disparity'][self.num_val:]

        dataset_root = os.path.join(self.root, 'dataset')

        train_disp = self._make_images(dataset_root, train_data, 'train')
        val_disp = self._make_images(dataset_root, val_data, 'validation')
        test_disp = self._make_images(dataset_root, test_data, 'test')

        self.disp['train'] = LITIV._make_gt(os.path.join(dataset_root, 'train'), train_disp)
        self.disp['validation'] = LITIV._make_gt(os.path.join(dataset_root, 'validation'), val_disp)
        self.disp['test'] = LITIV._make_gt(os.path.join(dataset_root, 'test'), test_disp)
