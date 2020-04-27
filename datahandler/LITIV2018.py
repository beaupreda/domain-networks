"""
information about the LITIV 2018 dataset.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import os
from collections import defaultdict
from typing import List

from datahandler.dataset import Dataset
from utils.enums import Datasets


class LITIV2018(Dataset):
    def __init__(self, root: str = None, psize: int = None, height: int = 480, width: int = 640, fold: int = None):
        """
        represents all information unique to LITIV 2018 dataset.
        :param root: path to the folder containing LITIV 2018 folder.
        :param psize: half size of the patch.
        :param height: image height.
        :param width: image width.
        """
        super(LITIV2018, self).__init__(root, psize, height, width, Datasets.litiv2018, fold)

        rgb, lwir, mrgb, mlwir, disparity, drange = self._prepare()
        mirrored = self._mirror(rgb, lwir, mrgb, mlwir, disparity, drange)
        self._add_points(rgb, lwir, mrgb, mlwir, disparity, drange, mirrored)

    def _prepare(self) -> (defaultdict[str, List[str]], defaultdict[str, List[str]], defaultdict[str, List[str]],
                           defaultdict[str, List[str]], defaultdict[str, List[str]], defaultdict[str, List[str]]):
        """
        aggregates all images and disparity files from the original LITIV 2018 folder.
        :return: void.
        """
        print(f'preparing LITIV 2018 dataset...')
        rgb_paths = defaultdict(list)
        lwir_paths = defaultdict(list)
        mask_rgb_paths = defaultdict(list)
        mask_lwir_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        drange_paths = defaultdict(list)
        videos = ['vid04', 'vid07', 'vid08']

        for video in videos:
            video_root = os.path.join(self.root, 'stcharles2018-v04', video)
            rectified_root = os.path.join(self.root, 'stcharles2018-v04', 'rectified', video)
            rgb_images = os.path.join(rectified_root, 'rgb')
            lwir_images = os.path.join(rectified_root, 'lwir')
            rgb = [r for r in os.listdir(rgb_images)]
            rgb.sort()
            lwir = [lw for lw in os.listdir(lwir_images)]
            lwir.sort()
            rgb = [os.path.join(rgb_images, r) for r in rgb]
            lwir = [os.path.join(lwir_images, lw) for lw in lwir]
            rgb_masks = os.path.join(video_root, 'rgb_gt_masks')
            mrgb = [r for r in os.listdir(rgb_masks)]
            mrgb.sort()
            mrgb = [os.path.join(rgb_masks, r) for r in mrgb]
            mrgb = mrgb[:len(rgb)]
            lwir_masks = os.path.join(video_root, 'lwir_gt_masks')
            mlwir = [lw for lw in os.listdir(lwir_masks)]
            mlwir.sort()
            mlwir = [os.path.join(lwir_masks, lw) for lw in mlwir]
            mlwir = mlwir[:len(lwir)]
            disparities = os.path.join(video_root, 'rgb_gt_disp')
            disparity = [d for d in os.listdir(disparities)]
            disparity.sort()
            disparity = [os.path.join(video_root, 'rgb_gt_disp', d) for d in disparity]
            drange = [os.path.join(video_root, 'drange.txt') for _ in range(len(rgb))]
            rgb_paths[video] = rgb
            lwir_paths[video] = lwir
            mask_rgb_paths[video] = mrgb
            mask_lwir_paths[video] = mlwir
            disparity_paths[video] = disparity
            drange_paths[video] = drange

        return self._reform(rgb_paths, lwir_paths, mask_rgb_paths, mask_lwir_paths, disparity_paths, drange_paths)
