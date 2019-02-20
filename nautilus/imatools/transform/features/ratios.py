from itertools import combinations

from cv2 import resize

import cv2
from numpy.core.multiarray import ndarray, array
from skimage.data import chelsea

from nautilus.transform.transform import Transform


class RatioBands(object):
    """"""
    
    def __init__(self,up_band: int, dwn_band: int):
        """Constructor for RatioBands"""
        self.up_band=up_band
        self.dwn_band = dwn_band
        

class Ratios(Transform):
    """"""

    def __init__(self,
                 n_bands: int=3
                 ):
        """Constructor for Ratios"""
        self.n_bands=n_bands
        self.ratios = list(
            map(
                lambda x: RatioBands(x[0], x[1]),
                combinations(range(n_bands), 2)
            )
        )

    def __call__(self, data: ndarray)->ndarray:
        assert len(data.shape)>2

        features = list()
        for ratio in self.ratios:
            features += (data[:, :, ratio.up_band] / data[:, :, ratio.dwn_band]).flatten().tolist()

        return array(features)

if __name__ == '__main__':
    im = chelsea()

    print(Ratios(n_bands=3)(im).shape)