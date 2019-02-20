from random import shuffle
from typing import Tuple, Iterator, Callable

from numpy.core.multiarray import ndarray
from numpy.ma import arange

from nautilus.data.dataset.sample_dataset import SampleDataset
from nautilus.data.sample.sample import Sample
from nautilus.data.sample.utils import SampleUtils
from nautilus.transform.identity import Identity
from nautilus.transform.transform import Transform


class Loader(object):
    """"""

    def __init__(self,
                 dataset: SampleDataset,
                 transform: Callable[[Sample], Sample]=Identity(),
                 batch_size: int=32,
                 n_epoch: int=10
                 ):
        """Constructor for Loader"""
        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size
        self.n_epoch=n_epoch

    def __iter__(self)->Iterator[Tuple[ndarray, ndarray]]:
        N = len(self.dataset)

        for _ in range(self.n_epoch):
            idxs = list(range(N))

            shuffle(idxs)

            limits = arange(0, N, self.batch_size)

            for start, stop in zip(limits[:-1], limits[1:]):
                selected_idxs = idxs[start:stop]

                samples = list(
                    map(
                        lambda idx: self.transform(self.dataset[idx]),
                        selected_idxs
                    )
                )

                yield SampleUtils.to_xy(samples)
