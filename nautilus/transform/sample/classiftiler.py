from typing import List

from numpy.ma import arange
from sklearn.feature_extraction.image import extract_patches_2d

from nautilus.data.sample.sample import Sample
from nautilus.transform.transform import Transform


class TileParameter(object):
    """"""

    def __init__(self,
                 tile_size: int,
                 stride: int,
                 ):
        """Constructor for TIleParameter"""
        self.tile_size = tile_size
        self.stride = stride


class ClassifTiler(Transform):
    """"""

    def __init__(self, tile_parameter: TileParameter):
        """Constructor for Tiler"""
        self.param = tile_parameter

    def __call__(self, sample: Sample)->List[Sample]:

        # samples = list()
        #
        # for i in arange(0, sample.x.shape[0] - self.param.tile_size + 1, self.param.stride):
        #     for j in arange(0, sample.x.shape[1] - self.param.tile_size + 1, self.param.stride):
        #         samples.append(
        #             Sample(
        #                 x=sample.x[i:i+self.param.tile_size, j:j+self.param.tile_size],
        #                 y=sample.y
        #             )
        #         )

        ext_patches = extract_patches_2d(sample.x, (self.param.tile_size, self.param.tile_size))[::self.param.stride,::]

        return list(map(lambda i: Sample(x=ext_patches[i,::], y=sample.y), range(ext_patches.shape[0])))





