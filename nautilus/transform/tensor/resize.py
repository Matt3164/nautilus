from cv2 import resize
from typing import Tuple

import cv2
from numpy.core.multiarray import ndarray

from nautilus.transform.transform import Transform


class Resize(Transform):
    """"""

    def __init__(self,
                 target_size: Tuple[int,int]):
        """Constructor for Resize"""
        self.target_size = target_size

    def __call__(self, data: ndarray):
        return resize(data, self.target_size, interpolation=cv2.INTER_CUBIC)


