from numpy.core.multiarray import ndarray
from skimage.data import chelsea
from skimage.feature import hog

from nautilus.transform.transform import Transform


class HOG(Transform):
    """"""

    def __init__(self,
                 orientations=8,
                 pixel_per_cell=16,
                 cells_per_block=1):
        """Constructor for HOG"""
        self.orientations=orientations
        self.pixels_per_cell = (pixel_per_cell, pixel_per_cell)
        self.cells_per_block=(cells_per_block,cells_per_block)

    def __call__(self, data: ndarray)->ndarray:
        out = hog(data, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=False, multichannel=True, feature_vector=True)

        return out


if __name__ == '__main__':
    im = chelsea()

    print(HOG()(im).shape)




