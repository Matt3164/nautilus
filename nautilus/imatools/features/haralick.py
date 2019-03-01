from typing import List

import cv2
from skimage.data import chelsea
from skimage.feature import greycomatrix, greycoprops
import numpy as np

from nautilus.transform.transform import Transform


class Haralick(Transform):
    """"""

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    def __init__(self,
                 distances: List[int]=[1],
                 angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                 levels: int=255
                 ):
        """Constructor for Haralick"""

        self.distances= distances
        self.angles = angles
        self.levels = levels

    def __call__(self, data: np.ndarray):

        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

        result = greycomatrix(data, self.distances, self.angles,
                              levels=self.levels)

        return np.concatenate([greycoprops(result, prop=prop) for prop in self.props], axis=1).flatten()


if __name__ == '__main__':
    # im = cv2.cvtColor(chelsea(), cv2.COLOR_RGB2GRAY)
    #
    # result = greycomatrix(im, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    #                       levels=255)
    #
    # print(result.shape)
    #
    #
    # props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    # print( np.concatenate([greycoprops(result, prop=prop) for prop in props], axis=1).flatten().shape )

    print( Haralick(distances=[1,2,4])(chelsea()).shape )
