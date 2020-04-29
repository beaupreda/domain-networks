"""
collection of useful functions.

author: David-Alexandre Beaupre
date: 2020-04-29
"""

import os
from argparse import Namespace
from typing import List

import cv2
import numpy as np
import torch


def extract_image_number(filename: str, from_file: bool) -> int:
    """
    finds the number of the image from the original images of LITIV 2014 dataset.
    :param filename: name of the image file.
    :param from_file: filename is read from a ground-truth file or not.
    :return: number.
    """
    number = filename[-8:-4]
    if from_file:
        number = filename[-9:-5]
    if not number.isdigit():
        number = number[-3:]
        if not number.isdigit():
            number = number[-2:]
    return int(number)


def is_patch_valid(points: (int, int, int), left: int, right: int, top: int, bottom: int) -> bool:
    """
    determines if a pair of patches centered on the points (x, y) and (dx, y) are valid.
    :param points: ground-truth points.
    :param left: lower bound (width).
    :param right: upper bound (width).
    :param top: lower bound (height).
    :param bottom: higher bound (height).
    :return: validity of the points.
    """
    x = int(points[0])
    y = int(points[1])
    dx = int(points[2])
    return (left < x < right) and (top < y < bottom) and (left < dx < right)


def special_sort(filenames: List[str]) -> List[str]:
    """
    sorts filenames according to their numbers, and not their string values.
    :param filenames: names of the file to sort.
    :return: sorted list of the filenames.
    """
    numbers = [extract_image_number(f, False) for f in filenames]
    idx = [i for i in range(len(filenames))]
    mapping = {}
    for i, n in zip(idx, numbers):
        mapping[n] = i
    numbers.sort()
    sorted_filenames = []
    for n in numbers:
        sorted_filenames.append(filenames[mapping[n]])
    return sorted_filenames


def mean_std_norm(img: np.ndarray) -> np.ndarray:
    """
    normalizes an image by subtracting its mean and dividing by its standard deviation.
    :param img: image to normalize.
    :return: normalized image.
    """
    return (img - np.mean(img, dtype=np.float32)) / np.std(img, dtype=np.float32)


def preprocess(img: np.ndarray, color: bool) -> np.ndarray:
    """
    preprocess operations on an image so it can be transformed to a torch tensor.
    :param img: image to preprocess.
    :param color: whether input image is color or grayscale.
    :return: preprocessed image.
    """
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, axes=(2, 0, 1))
    img = mean_std_norm(img)
    return img


def create_positive_negative_samples(disparity: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """
    creates matching and non-matching sample points from the dataset.
    :param disparity: ground-truth data points.
    :param positive: range of positive offsets.
    :param negative: range of negative offsets.
    :return: balanced ground-truth data points (as many positive samples as negative ones).
    """
    # positive samples
    num_samples = disparity.shape[0]
    labels = np.ones(shape=(num_samples, 1), dtype=np.int32)
    disparity = np.hstack((disparity, labels))
    offsets = np.random.choice(positive, size=num_samples).reshape(num_samples)
    disparity[:, 3] = np.add(disparity[:, 3], offsets)

    # negative samples
    disparity = np.vstack((disparity.copy(), disparity.copy()))
    labels = np.zeros_like(labels)
    offsets = np.random.choice(negative, size=num_samples).reshape(num_samples)
    disparity[num_samples:, 3] = np.add(disparity[num_samples:, 3], offsets)
    disparity[num_samples:, 4] = labels.squeeze()

    # shuffle everything
    shuffled_idx = np.arange(disparity.shape[0], dtype=np.int32)
    np.random.shuffle(shuffled_idx)
    disparity = disparity[shuffled_idx]

    return disparity


def get_number_parameters(model: torch.nn.Module) -> int:
    """
    counts the number of trainable parameters in the given model.
    :param model: torch model.
    :return: number of parameters.
    """
    return sum([p.data.nelement() for p in model.parameters()])


def save_model(savepath: str, model: torch.nn.Module, epoch: int, train_loss: float, validation_loss: float) -> None:
    """
    saves relevant information during the training phase.
    :param savepath: path to save file.
    :param model: torch model.
    :param epoch: current epoch.
    :param train_loss: current training loss.
    :param validation_loss: current validation loss.
    :return: void.
    """
    print('saving model...')
    filename = os.path.join(savepath, 'parameters', f'params{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'train_loss': train_loss,
        'validation_loss': validation_loss
    }, filename)


def adjust_learning_rate(optimizer: torch.optim, args: Namespace) -> None:
    """
    sets a new learning rate for the optimizer.
    :param optimizer: torch optimizer.
    :param args: structure containing all arguments.
    :return: void.
    """
    learning_rate = args.learning_rate / 2.0
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = learning_rate
    print(f'new learning rate is {learning_rate:.4f}')
