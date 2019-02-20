from typing import List, Tuple

from numpy.core._multiarray_umath import ndarray
from numpy.ma import array

from nautilus.data.sample.sample import Sample
from nautilus.transform.transform import Transform


class SampleToArrays(Transform):
    """"""

    def __call__(self, samples: List[Sample])-> Tuple[ndarray, ndarray]:
        return array(list(map(lambda sample: sample.x, samples))), array(list(map(lambda sample: sample.y, samples)))

samples_to_xy=SampleToArrays()



