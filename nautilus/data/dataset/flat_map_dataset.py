from typing import Callable, List

from nautilus.data.dataset.dataset import Dataset
from nautilus.data.sample.sample import Sample


class FlatMapDataset(Dataset):
    """"""

    def __init__(self, dataset: Dataset, trs: Callable[[Sample], List[Sample]]):
        """Constructor for FlatMapDataset"""
        self.dataset = dataset
        self.trs = trs

        self.n_sample_by_sample = len(trs(dataset[0]))

    def __len__(self):
        return len(self.dataset)*self.n_sample_by_sample

    def __getitem__(self, item)->Sample:

        sample_idx = item // self.n_sample_by_sample

        origin_sample = self.dataset[sample_idx]

        sample_idx = item % self.n_sample_by_sample

        return self.trs(origin_sample)[sample_idx]






