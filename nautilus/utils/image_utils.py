from cv2 import resize
from functools import partial
from typing import Tuple, Callable

import cv2

from nautilus.utils import np_utils
from sklearn.feature_extraction.image import extract_patches_2d

def cvresize(arr: np_utils.ndarray, target_size: Tuple[int,int])->np_utils.ndarray:
    return resize(arr, target_size, interpolation=cv2.INTER_CUBIC)


def cvresize_fn(target_size: Tuple[int, int])->Callable[[np_utils.ndarray], np_utils.ndarray]:
    return partial(cvresize, target_size=target_size)


