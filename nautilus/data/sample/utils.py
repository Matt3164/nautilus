from typing import List, Tuple, Callable

from numpy.core.multiarray import ndarray

from nautilus.data.sample.sample import Sample
from nautilus.transform.sample.to_array import SampleToArrays


class SampleUtils(object):
    """"""

    @staticmethod
    def to_xy(samples: List[Sample])->Tuple[ndarray, ndarray]:
        return SampleToArrays()(samples)