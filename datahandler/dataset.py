"""
base class for the LITIV 2014 and 2018 datasets.

author: David-Alexandre Beaupre
date: 2020-04-22
"""

import glob
import os
from collections import defaultdict
from typing import Dict, DefaultDict, List

import cv2
import numpy as np

import utils.io as io
import utils.misc as misc
from utils.enums import Datasets


class Dataset:
    def __init__(self, root: str = None, psize: int = None, height: int = 0, width: int = 0,
                 dataset: Datasets = None, fold: int = 0):
        """
        represents common structure for both LITIV 2014 and LITIV 2018 datasets.
        :param root: path to the folder containing both LITIV datasets folders.
        :param psize: half size of the patch.
        :param height: image height.
        :param width: image width.
        :param dataset: enum value (see enums.py in utils).
        """
        self.root = root
        self.psize = psize
        self.height = height
        self.width = width
        self.dataset = dataset
        self.fold = fold

        self.rgb = defaultdict(list)
        self.lwir = defaultdict(list)
        self.mask_rgb = defaultdict(list)
        self.mask_lwir = defaultdict(list)
        self.disparity = defaultdict(list)

        assert self.root is not None, 'please specify the root of the LITIV dataset.'
        assert self.psize > 0, 'psize must be > 0.'
        assert self.height > 0, 'image height must be > 0.'
        assert self.width > 0, 'image width must be > 0.'
        assert self.dataset == Datasets.LITIV2014 or self.dataset == Datasets.LITIV2018, 'unknown dataset.'
        assert 0 < self.fold < 7, 'possible fold values are: 1, 2, 3, 4, 5 and 6.'

        # TODO: verify that.
        self.dataset_root = os.path.join(self.root, 'dataset')
        if not os.path.isdir(self.dataset_root):
            os.mkdir(self.dataset_root)

    def _prepare(self):
        """
        see class implementations.
        :return: void.
        """
        pass

    def _reform(self, rgb: DefaultDict[str, List[str]], lwir: DefaultDict[str, List[str]],
                mrgb: DefaultDict[str, List[str]], mlwir: DefaultDict[str, List[str]],
                disparity: DefaultDict[str, List[str]], drange: DefaultDict[str, List[str]]) -> \
            (DefaultDict[str, List[str]], DefaultDict[str, List[str]], DefaultDict[str, List[str]],
             DefaultDict[str, List[str]], DefaultDict[str, List[str]], DefaultDict[str, List[str]]):
        """
        creates two new folders so both datasets have the same structure, which will facilitates the merging later on.
        :param rgb: dict(str, list) of paths to original rgb images.
        :param lwir: dict(str, list) of paths to original lwir images.
        :param mrgb: dict(str, list) of paths to original rgb masks.
        :param mlwir: dict(str, list) of paths to original lwir mask.
        :param disparity: dict(str, list) of paths to original disparity ground-truth files.
        :param drange: dict(str, list) of paths to original disparity ranges files.
        :return: six dict(str, list) for each video with the paths to images, masks, disparity and drange files.
        """
        names = ['LITIV2014', 'LITIV2018']
        name = names[self.dataset]
        print(f'reforming {name} dataset...')
        reformed_root = os.path.join(self.root, name)
        if not os.path.isdir(reformed_root):
            os.mkdir(reformed_root)

        rgb_images = defaultdict(list)
        lwir_images = defaultdict(list)
        masks_rgb = defaultdict(list)
        masks_lwir = defaultdict(list)
        disparity_files = defaultdict(list)
        drange_files = defaultdict(list)

        for video in rgb.keys():
            video_root = os.path.join(reformed_root, video)
            if not os.path.isdir(video_root):
                os.mkdir(video_root)

            rgb_root = os.path.join(video_root, 'rgb')
            if not os.path.isdir(rgb_root):
                os.mkdir(rgb_root)
            # remove all rgb images, they will be recreated
            files = glob.glob(os.path.join(rgb_root, '*'))
            for file in files:
                os.remove(file)

            lwir_root = os.path.join(video_root, 'lwir')
            if not os.path.isdir(lwir_root):
                os.mkdir(lwir_root)
            # remove all lwir images, they will be recreated
            files = glob.glob(os.path.join(lwir_root, '*'))
            for file in files:
                os.remove(file)

            rgb_masks_root = os.path.join(video_root, 'masks_rgb')
            if not os.path.isdir(rgb_masks_root):
                os.mkdir(rgb_masks_root)
            # remove all rgb masks images, they will be recreated
            files = glob.glob(os.path.join(rgb_masks_root, '*'))
            for file in files:
                os.remove(file)

            lwir_masks_root = os.path.join(video_root, 'masks_lwir')
            if not os.path.isdir(lwir_masks_root):
                os.mkdir(lwir_masks_root)
            # remove all lwir masks images, they will be recreated
            files = glob.glob(os.path.join(lwir_masks_root, '*'))
            for file in files:
                os.remove(file)

            disparity_root = os.path.join(video_root, 'disparity')
            if not os.path.isdir(disparity_root):
                os.mkdir(disparity_root)
            # remove all disparity files, they will be recreated
            files = glob.glob(os.path.join(disparity_root, '*'))
            for file in files:
                os.remove(file)

            drange_root = os.path.join(video_root, 'drange')
            if not os.path.isdir(drange_root):
                os.mkdir(drange_root)
            # remove all drange files, they will be recreated
            files = glob.glob(os.path.join(drange_root, '*'))
            for file in files:
                os.remove(file)

            visited = set()
            idx = 0
            for i, (r, l, mr, ml, d, dr) in enumerate(zip(rgb[video], lwir[video], mrgb[video], mlwir[video],
                                                          disparity[video], drange[video])):
                # copy files as i.extension (m[-3:] = last 3 characters of the string).
                rgb_images[video].append(io.copy_image(r, os.path.join(rgb_root, f'{i}.{r[-3:]}')))
                lwir_images[video].append(io.copy_image(l, os.path.join(lwir_root, f'{i}.{l[-3:]}')))
                masks_rgb[video].append(io.copy_image(mr, os.path.join(rgb_masks_root, f'{i}.{mr[-3:]}')))
                masks_lwir[video].append(io.copy_image(ml, os.path.join(lwir_masks_root, f'{i}.{ml[-3:]}')))
                if d not in visited:
                    j, new_disparity = io.unify_ground_truth(d, disparity_root, i, self.dataset, idx)
                    if self.dataset == Datasets.LITIV2014:
                        disparity_files[video].extend(new_disparity)
                    else:
                        disparity_files[video].append(new_disparity)
                    idx += j
                    visited.add(d)
                drange_files[video].append(io.copy_file(dr, os.path.join(drange_root, f'{i}.{dr[-3:]}')))

        return rgb_images, lwir_images, masks_rgb, masks_lwir, disparity_files, drange_files

    def _mirror(self, rgb: DefaultDict[str, List[str]], lwir: DefaultDict[str, List[str]],
                mrgb: DefaultDict[str, List[str]], mlwir: DefaultDict[str, List[str]],
                disparity: DefaultDict[str, List[str]], drange: DefaultDict[str, List[str]]) -> Dict[str, int]:
        """
        duplicates every image with a left right flip to augment the dataset.
        :param rgb: dict(str, list) of paths to reformed rgb images.
        :param lwir: dict(str, list) of paths to reformed lwir images.
        :param mrgb: dict(str, list) of paths to reformed rgb masks.
        :param mlwir: dict(str, list) of paths to reformed lwir masks.
        :param disparity: dict(str, list) of paths to reformed disparity ground-truth files.
        :param drange: dict(str, list) of paths to reformed disparity ranges files.
        :return: dict(str, int) of the index of the first mirrored image for each video.
        """
        extensions = ['.jpg', '.png']
        names = ['LITIV2014', 'LITIV2018']
        name = names[self.dataset]
        print(f'mirroring frames from {name} dataset...')
        mirrored = {}
        for video in rgb.keys():
            assert len(rgb[video]) == len(lwir[video]) == len(mrgb[video]) == len(mlwir[video]) == \
                   len(disparity[video]) == len(drange[video]), 'lists must be of the same length.'

            video_root = os.path.join(self.root, name, video)
            rgb_root = os.path.join(video_root, 'rgb')
            lwir_root = os.path.join(video_root, 'lwir')
            rgb_masks_root = os.path.join(video_root, 'masks_rgb')
            lwir_masks_root = os.path.join(video_root, 'masks_lwir')
            disparity_root = os.path.join(video_root, 'disparity')
            drange_root = os.path.join(video_root, 'drange')

            num_elements = len(rgb[video])
            mirrored[video] = num_elements
            rgb_images = []
            lwir_images = []
            masks_rgb = []
            masks_lwir = []
            disparity_files = []
            drange_files = []
            for i, (r, l, mr, ml, d, dr) in enumerate(zip(rgb[video], lwir[video], mrgb[video], mlwir[video],
                                                          disparity[video], drange[video])):
                name_rgb = os.path.join(rgb_root, f'{num_elements + i}{extensions[self.dataset]}')
                name_lwir = os.path.join(lwir_root, f'{num_elements + i}{extensions[self.dataset]}')
                name_mask_rgb = os.path.join(rgb_masks_root, f'{num_elements + i}{extensions[self.dataset]}')
                name_mask_lwir = os.path.join(lwir_masks_root, f'{num_elements + i}{extensions[self.dataset]}')
                name_disparity = os.path.join(disparity_root, f'{num_elements + i}.txt')
                name_drange = os.path.join(drange_root, f'{num_elements + i}.txt')
                cv2.imwrite(name_rgb, np.fliplr(cv2.imread(r, cv2.IMREAD_COLOR)))
                cv2.imwrite(name_lwir, np.fliplr(cv2.imread(l, cv2.IMREAD_COLOR)))
                cv2.imwrite(name_mask_rgb, np.fliplr(cv2.imread(mr, cv2.IMREAD_COLOR)))
                cv2.imwrite(name_mask_lwir, np.fliplr(cv2.imread(ml, cv2.IMREAD_COLOR)))
                io.copy_file(d, name_disparity)
                io.copy_file(dr, name_drange)
                rgb_images.append(name_rgb)
                lwir_images.append(name_lwir)
                masks_rgb.append(name_mask_rgb)
                masks_lwir.append(name_mask_lwir)
                disparity_files.append(name_disparity)
                drange_files.append(name_drange)

            rgb[video].extend(rgb_images)
            lwir[video].extend(lwir_images)
            mrgb[video].extend(masks_rgb)
            mlwir[video].extend(masks_lwir)
            disparity[video].extend(disparity_files)
            drange[video].extend(drange_files)

        return mirrored

    def _add_points(self, rgb: DefaultDict[str, List[str]], lwir: DefaultDict[str, List[str]],
                    mrgb: DefaultDict[str, List[str]], mlwir: DefaultDict[str, List[str]],
                    disparity: DefaultDict[str, List[str]], drange: DefaultDict[str, List[str]],
                    mirrored: Dict[str, int]) -> None:
        """
        add data points below (x, y - 1) and above (x, y + 1) for all know ground-truth disparities (x, y)
        :param rgb: dict(str, list) of paths to rgb images.
        :param lwir: dict(str, list) of paths to lwir images.
        :param mrgb: dict(str, list) of paths to rgb masks.
        :param mlwir: dict(str, list) of paths to lwir masks.
        :param disparity: dict(str, list) of paths to disparity ground-truth files.
        :param drange: dict(str, list) of paths to disparity ranges files.
        :param mirrored: dict(str, int) of the index of the first mirrored image for each video.
        :return: void.
        """
        print(f'adding training points...')
        left = self.psize + self.psize
        right = self.width - (self.psize + self.psize + 1)
        top = self.psize
        bottom = self.height - (self.psize + 1)

        for video in disparity.keys():
            m = mirrored[video]
            for i, (r, l, mr, ml, d, dr) in enumerate(zip(rgb[video], lwir[video], mrgb[video], mlwir[video],
                                                          disparity[video], drange[video])):
                points = io.read_disparity_file(d, dr, i, m, self.width, self.dataset)
                # add data points of 4 neighbors: up, down, left, right
                upoints = np.hstack((points[:, 0], points[:, 1] + 1, points[:, 2])).reshape(points.shape, order='F')
                dpoints = np.hstack((points[:, 0], points[:, 1] - 1, points[:, 2])).reshape(points.shape, order='F')
                lpoints = np.hstack((points[:, 0] - 1, points[:, 1], points[:, 2])).reshape(points.shape, order='F')
                rpoints = np.hstack((points[:, 0] + 1, points[:, 1], points[:, 2])).reshape(points.shape, order='F')
                points = np.vstack((points, upoints, dpoints, lpoints, rpoints))
                with open(d, 'w') as file:
                    for j in range(points.shape[0]):
                        if misc.is_patch_valid(points[j, :], left, right, top, bottom):
                            file.write(f'{int(points[j, 0])} {int(points[j, 1])} {int(points[j, 2])}\n')
                self.rgb[video].append(r)
                self.lwir[video].append(l)
                self.mask_rgb[video].append(mr)
                self.mask_lwir[video].append(ml)
                self.disparity[video].append(d)
