from skimage.data import chelsea

from nautilus.transform.transform import Transform
# import the necessary packages
from skimage import feature
import numpy as np


class LBP(Transform):
    """"""

    def __init__(self,numPoints: int, radius: int, eps=1e-7):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.eps=eps

    def __call__(self, data: np.ndarray):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(data, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + self.eps)

        # return the histogram of Local Binary Patterns
        return hist

    @staticmethod
    def from_radius(radius: int):
        return LBP(numPoints=8*radius, radius=radius)

if __name__ == '__main__':
    radius=5
    numPoints=8*radius
    print(LBP(numPoints=numPoints, radius=radius)(chelsea()[:,:,0]).shape)