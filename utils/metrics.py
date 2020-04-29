"""
metrics used.

author: David-Alexandre Beaupre
date: 2020-04-29
"""

import torch


def correct_matches_distance_n(predictions: torch.Tensor, targets: torch.Tensor, n: int = 3) -> torch.Tensor:
    """
    computes the number of correct matches at a distance <= of n to the ground-truth.
    :param predictions: tensor containing the predicted disparity values.
    :param targets: tensor containing the disparity values.
    :param n: distance to ground-truth that is considered correct (lower or equal).
    :return: number of correct matches.
    """
    assert predictions.size() == targets.size(), 'prediction and target tensors must be of the same size.'
    if n < 0:
        print(f'n = {n}. It must be 0 or greater. Setting it to 0 for this function call.')
        n = 0
    return torch.sum(torch.le(torch.abs(predictions - targets), n))


def n_pixel_error(predictions: torch.Tensor, targets: torch.Tensor, n: int = 3) -> float:
    """
    computes the n-pixel error metric (ratio of pixels at distance > n of ground-truth disparity).
    :param predictions: tensor containing the predicted disparity values.
    :param targets: tensor containing the disparity values.
    :param n: distance to ground-truth that is considered correct (lower or equal).
    :return: error ratio.
    """
    correct = correct_matches_distance_n(predictions, targets, n)
    return 1.0 - (float(correct) / float(predictions.numel()))
