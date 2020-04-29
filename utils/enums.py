"""
relevant enums.

author: David-Alexandre Beaupre
date: 2020-04-22
"""

from enum import IntEnum


class Datasets(IntEnum):
    """
    choice of the LITIV dataset.
    """
    LITIV2014 = 0,
    LITIV2018 = 1
