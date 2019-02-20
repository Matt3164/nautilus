from nautilus.data.dataset.dataset import Dataset
from nautilus.data.sample.sample import Sample


class SampleDataset(Dataset):
    """"""

    def __getitem__(self, item)->Sample:
        raise NotImplementedError


