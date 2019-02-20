import cv2
from skimage.data import chelsea
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
import numpy as np

from nautilus.transform.transform import Transform


class Haar(Transform):
    """"""

    def __init__(self, ):
        """Constructor for Haar"""
        pass

    def __call__(self, data: np.ndarray):


        if len(data.shape)>2:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

        ii = integral_image(data)

        features = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

        return haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1],
            feature_type=features,
        )


if __name__ == '__main__':
    im = chelsea()[:, :, 0]

    ii = integral_image(im)

    features = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

    print(haar_like_feature(ii, 0, 0, 8, 8,
                            feature_type=features,
                            ).shape)
