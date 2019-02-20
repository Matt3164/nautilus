from typing import Callable

from numpy.core.multiarray import ndarray

from nautilus.data.sample.sample import Sample
from nautilus.transform.identity import Identity
from nautilus.transform.transform import Transform


class SampleTransform(Transform):
    """"""

    def __init__(self,
                 x_trs: Callable[[ndarray], ndarray]=Identity(),
                 y_trs: Callable[[ndarray], ndarray]=Identity()
                 ):
        """Constructor for SampleTransform"""

        self.x_trs = x_trs
        self.y_trs = y_trs

    def __call__(self, sample: Sample)->Sample:
        return Sample(
            x=self.x_trs(sample.x),
            y=self.y_trs(sample.y),
        )


