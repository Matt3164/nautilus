from typing import Tuple

import cv2
from numpy.core.multiarray import ndarray
import numpy as np
from nautilus.transform.transform import Transform


class Hist(Transform):
    """"""

    def __init__(self,
                 n_bands: int,
                 bins_by_band: int,
                 band_range: Tuple[int,int]=(0,256)) -> None:
        super().__init__()
        self.n_bands=n_bands
        self.bands=range(self.n_bands)
        self.bins_by_band = bins_by_band
        self.band_range = band_range


    def __call__(self, data: ndarray):
        a = [b for band in self.bands for b in self.band_range]
        return cv2.calcHist([data], list(self.bands), None, [self.bins_by_band]*self.n_bands, [b for band in self.bands for b in self.band_range]).flatten()


if __name__ == '__main__':
    print(Hist(n_bands=3, bins_by_band=8, band_range=(0,256))(np.random.randint(0, 255, size=(256, 256, 3)).astype(np.uint8)).shape)

    print(cv2.calcHist([np.random.randint(0, 255, size=(256, 256, 3)).astype(np.uint8)], list(range(3)), None, [8] * 3,
                 [b for band in range(3)for b in (0,256)]).shape)