from typing import Callable

from nautilus.data.dataset.dataset import Dataset
from nautilus.data.dataset.utils import DatasetUtils
from nautilus.data.sample.sample import Sample
from nautilus.data.sample.utils import SampleUtils
from nautilus.transform.transform import Transform


class DatasetToArrayDataset(Transform):
    """"""

    def __init__(self, transform: Callable[[Sample], Sample]=lambda x: x, subsample: float=None):
        self.trs = transform
        self.subsample = subsample

    def __call__(self, data: Dataset)->Dataset:

        new_dataset = DatasetUtils.from_dataset_and_trs(data, transform=self.trs)

        if self.subsample is None:
            n_max = len(data)
        else:
            n_max = int( len(data)*self.subsample )

        samples = list()
        for i in range(n_max):
            samples.append(new_dataset[i])

        X, Y = SampleUtils.to_xy(samples)

        return DatasetUtils.from_xy_array(X, Y)





