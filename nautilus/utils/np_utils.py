from typing import Callable, List

from numpy import load, ndarray, savez_compressed
from numpy import array as nparray

from itertools import chain

def save_npz(filepath: str, array_dict):
    return savez_compressed(filepath, **array_dict)

def map_arr(array: ndarray, func: Callable[[ndarray], ndarray])->ndarray:
    return nparray(list(map(func, array)))

def flatmap_arr(array: ndarray, func: Callable[[ndarray], List[ndarray]])->ndarray:
    return nparray(list(chain.from_iterable( map(func, array) ) ))

def flatten(arr: ndarray):
    return arr.flatten()