from typing import Tuple

from numpy.core.multiarray import ndarray

from nautilus.data.dataset.dataset import Dataset
from nautilus.data.sample.utils import SampleUtils
from nautilus.transform.transform import Transform


class DatasetToTensor(Transform):
    """"""

    def __call__(self, dataset: Dataset)->Tuple[ndarray, ndarray]:
        data = list()

        for i in range(len(dataset)):
            data.append(dataset[i])

        return SampleUtils.to_xy(data)


