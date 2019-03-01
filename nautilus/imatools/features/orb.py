import cv2
from numpy.core.multiarray import ndarray
from skimage.data import chelsea
from skimage.feature import ORB

from nautilus.transform.transform import Transform


class OrbDescriptor(Transform):
    """"""

    def __init__(self,
                 n_scales=8,
                 n_keypoints=500
                 ):
        """Constructor for OrbDescriptor"""
        self.n_scales = n_scales
        self.n_keypoints = n_keypoints

    def __call__(self, data: ndarray)->ndarray:

        orb = ORB(n_keypoints=self.n_keypoints, n_scales=self.n_scales)

        if len(data.shape)>2:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

        orb.detect_and_extract(data)

        return orb.descriptors


if __name__ == '__main__':
    im = chelsea()

    # orb = ORB(n_keypoints=10)
    #
    # orb.detect_and_extract(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
    #
    # print(orb.keypoints.shape)
    #
    # print(orb.descriptors.shape)

    print( OrbDescriptor(n_keypoints=100)(im).shape )