from numpy.core.multiarray import ndarray

from nautilus.data.dataset.dataset import Dataset


class TensorDataset(Dataset):
    """"""

    def __getitem__(self, item)->ndarray:
        raise NotImplementedError


