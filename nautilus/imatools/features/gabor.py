from itertools import chain
from typing import List

import numpy as np
from skimage.data import chelsea
from skimage.filters import gabor_kernel
from scipy import ndimage as nd

from nautilus.transform.transform import Transform


class SingleBandGabor(Transform):
    """"""

    def __init__(self,
                 theta: int,
                 sigma: List[float],
                 frequency: List[float]):
        """Constructor for SingleBandGabor"""
        self.theta=theta
        self.sigma=sigma
        self.frequency=frequency

        self.kernels = list()

        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    self.kernels.append(kernel)

    def __call__(self, data: np.ndarray):
        feats = np.zeros((len(self.kernels), 2), dtype=np.double)
        for k, kernel in enumerate(self.kernels):
            filtered = nd.convolve(data, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()

        return feats.flatten()

class Gabor(Transform):
    """"""

    def __init__(self,
                 theta: int,
                 sigma: List[float],
                 frequency: List[float],
                 n_bands: int):
        """Constructor for SingleBandGabor"""
        self.theta = theta
        self.sigma = sigma
        self.frequency = frequency
        self.n_bands = n_bands

        self.singles=list()
        for b in range(n_bands):
            self.singles.append(SingleBandGabor(theta=theta, sigma=sigma, frequency=frequency))


    def __call__(self, data: np.ndarray):
        features = [self.singles[b](data[:,:,b]) for b in range(data.shape[-1])]

        features = map(lambda x: x.tolist(), features)

        return np.array( list(chain.from_iterable(features)) )


if __name__ == '__main__':

    im = chelsea()

    # prepare filter bank kernels
    # kernels = []
    #
    # for theta in range(4):
    #     theta = theta / 4. * np.pi
    #     for sigma in (1, 3):
    #         for frequency in (0.05, 0.25):
    #             kernel = np.real(gabor_kernel(frequency, theta=theta,
    #                                           sigma_x=sigma, sigma_y=sigma))
    #             kernels.append(kernel)
    #
    #
    # feats = np.zeros((len(kernels), 2), dtype=np.double)
    # for k, kernel in enumerate(kernels):
    #     filtered = nd.convolve(im, kernel, mode='wrap')
    #     feats[k, 0] = filtered.mean()
    #     feats[k, 1] = filtered.var()
    #
    # print(feats.shape)
    #
    #

    # print(SingleBandGabor(theta=4, sigma=[1,3], frequency=[.05, .25])(im).shape)
    print(Gabor(theta=4, sigma=[1, 3], frequency=[.05, .25], n_bands=3)(im).shape)
