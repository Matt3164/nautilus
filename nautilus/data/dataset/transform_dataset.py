from typing import Callable

from nautilus.data.dataset.sample_dataset import SampleDataset
from nautilus.data.sample.sample import Sample


class TransformDataset(SampleDataset):
    """"""

    def __init__(self,
                 dataset: SampleDataset,
                 transform: Callable[[Sample], Sample]
                 ):
        """Constructor for TransformDataset"""
        self.origin_dataset = dataset
        self.trs = transform

    def __getitem__(self, item) -> Sample:

        return self.trs(self.origin_dataset[item])

    def __len__(self):
        return len(self.origin_dataset)



