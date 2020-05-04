"""
handles reading and writing different types of files.

author: David-Alexandre Beaupre
date: 2020-04-28
"""

import os
from collections import defaultdict
from typing import List, Tuple, DefaultDict

import cv2
import numpy as np
import yaml

import utils.misc as misc
from utils.enums import Datasets


# TODO: rename read GT files
def copy_image(old_location: str, new_location: str) -> str:
    """
    copy images from one location to another.
    :param old_location: path to original image.
    :param new_location: path to new image.
    :return: new path location.
    """
    cv2.imwrite(new_location, cv2.imread(old_location, cv2.IMREAD_COLOR))
    return new_location


def copy_file(old_location: str, new_location: str) -> str:
    """
    copy the content of a file from one location to another.
    :param old_location: path to original file.
    :param new_location: path to new file.
    :return: new path location.
    """
    with open(old_location, 'r') as o:
        lines = o.readlines()
    with open(new_location, 'w') as n:
        n.writelines(lines)
    return new_location


def unify_ground_truth(old_filename: str, disparity_root: str, i: int, dataset: Datasets, idx: int) -> \
        (int, List[str]) or (int, str):
    """
    reads ground-truth files from LITIV 2014 and 2018 datasets and writes new ground-truth files in a uniformed way.
    :param old_filename: path to the original file.
    :param disparity_root: path to disparity folder.
    :param i: index for LITIV 2018.
    :param dataset: which LITIV dataset.
    :param idx: offset for LITIV 2014.
    :return: offset and filename(s).
    """
    points = read_ground_truth_disparities(old_filename, dataset)
    return write_ground_truth_disparities(disparity_root, points, i, dataset, idx)


def read_drange(filename: str) -> (int, int):
    """
    read drange file.
    :param filename: name of the drange file.
    :return: low and high range values.
    """
    with open(filename, 'r') as file:
        line = file.readline()
        line = line.split(' ')
    return int(line[0]), int(line[1])


def read_disparity(filename: str, i: int) -> List[Tuple[int, int, int, int]]:
    """
    reads content of disparity file.
    :param filename: name of GT file.
    :param i: index of file.
    :return: data points.
    """
    points = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.split(' ')
            x = int(line[0])
            y = int(line[1])
            dx = int(line[2])
            points.append((i, x, y, dx))
    return points


def read_disparity_file(filename: str, drange: str, i: int, m: int, width: int, dataset: Datasets) -> np.ndarray:
    """
    reads the data points and corrects their position with the range.
    :param filename: name of disparity file.
    :param drange: name of drange file.
    :param i: current file index.
    :param m: mirror file index.
    :param width: width of the images in the dataset.
    :param dataset: which LITIV dataset.
    :return: data points in the corrected range.
    """
    points = []
    if dataset == Datasets.LITIV2014:
        high = 0
    else:
        _, high = read_drange(drange)
    with open(filename, 'r') as file:
        for line in file:
            line = line.split(' ')
            x = int(line[0])
            y = int(line[1])
            d = int(line[2])
            if dataset == Datasets.LITIV2014:
                dx = x
                x = x - abs(d)
                if i >= m:
                    dx = width - dx
                    x = width - x
            else:
                dx = x + d + high
                if i < m:
                    x = width - x
                    dx = width - dx
            points.append([x, y, dx])
    return np.array(points, dtype=np.int32)


def read_disparity_gt(filename: str) -> np.ndarray:
    """
    reads the disparity files used for training/testing.
    :param filename: name of the file.
    :return: data points.
    """
    points = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.split(' ')
            frame = int(line[0])
            x_rgb = int(line[1])
            y = int(line[2])
            x_ir = int(line[3])
            points.append([frame, x_rgb, y, x_ir])
    return np.array(points, dtype=np.int32)


def read_ground_truth_disparities(filename: str, dataset: Datasets) -> \
        DefaultDict[int, List[Tuple[int, int, int]]] or List[Tuple[int, int, int]]:
    """
    reads the original ground-truth files of LITIV 2014 and 2018 datasets.
    :param filename: name of ground-truth file.
    :param dataset: which LITIV dataset.
    :return: data points for each frame.
    """
    if dataset == Datasets.LITIV2014:
        points = defaultdict(list)
        with open(filename, 'r') as file:
            while True:
                lwir = file.readline()
                rgb = file.readline()
                x = file.readline()
                y = file.readline()
                d = file.readline()
                if lwir != '':
                    number = misc.extract_image_number(rgb, True)
                    points[number].append((int(x), int(y), int(d)))
                else:
                    break
            return points
    elif dataset == Datasets.LITIV2018:
        points = []
        with open(filename, 'r') as file:
            # # ignore first line (%YAML:1.0)
            _ = file.readline()
            _ = file.readline()
            disp_info = yaml.safe_load(file)
            for key in disp_info.keys():
                if type(disp_info[key]) is dict:
                    x = disp_info[key]['x']
                    y = disp_info[key]['y']
                    d = disp_info[key]['d']
                    points.append((int(x), int(y), int(d)))
            return points


def write_ground_truth_disparities(disparity_root: str,
                                   points: DefaultDict[int, List[Tuple[int, int, int]]] or List[Tuple[int, int, int]],
                                   i: int, dataset: Datasets, idx: int) -> (int, List[str]) or Tuple[int, str]:
    """
    writes ground-truth data in a uniformed manner for both datasets.
    :param disparity_root: path to disparity folder.
    :param points: data points.
    :param i: index for LITIV 2018.
    :param dataset: which LITIV dataset.
    :param idx: offset for LITIV 2014.
    :return: offset and filename(s).
    """
    if dataset == Datasets.LITIV2014:
        filenames = []
        for i, frame in enumerate(points.keys()):
            filename = os.path.join(disparity_root, f'{idx + i}.txt')
            with open(filename, 'w') as file:
                for x, y, d in points[frame]:
                    file.write(f'{x} {y} {d}\n')
                filenames.append(filename)
        return len(points), filenames
    elif dataset == Datasets.LITIV2018:
        filename = os.path.join(disparity_root, f'{i}.txt')
        with open(filename, 'w') as file:
            for x, y, d in points:
                file.write(f'{x} {y} {d}\n')
        return 0, filename
