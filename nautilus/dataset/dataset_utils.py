from itertools import chain
from typing import Tuple, Callable, List, Iterator

from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split

from nautilus.dataset.dataset import Dataset
from nautilus.dataset.train_test_dataset import TrainTestDataset
from nautilus.transform.transform import Transform
from nautilus.utils import np_utils


def split(dataset: Dataset, ratio: float) -> TrainTestDataset:
    xtrain, xtest, ytrain, ytest = train_test_split(dataset.X, dataset.Y,
                                                    train_size=ratio,
                                                    test_size=1. - ratio)

    return TrainTestDataset(
        train_dataset=Dataset.from_arrays(xtrain, ytrain),
        test_dataset=Dataset.from_arrays(xtest, ytest)
    )

def to_iterator(dataset: Dataset) -> Iterator[Tuple[ndarray, ndarray]]:
    for i in range(dataset.X.shape[0]):
        yield dataset.X[i, ::], dataset.Y[i, :]


def from_iterator(iterator: Iterator[Tuple[ndarray, ndarray]]) -> Dataset:
    l = list(iterator)

    return Dataset.from_arrays(
        X=np_utils.nparray(list(map(lambda x: x[0], l))),
        Y=np_utils.nparray(list(map(lambda x: x[1], l)))
    )


def map_x(dataset: Dataset, map_fn: Callable[[ndarray], ndarray]) -> Dataset:
    return Dataset.from_arrays(
        np_utils.map_arr(dataset.X, map_fn),
        dataset.Y
    )

# def flatmap(dataset: Dataset, arr_func: Callable[[Tuple[ndarray, ndarray]],
#                                                  Iterator[Tuple[ndarray,
#                                                                 ndarray]]]):
#     xy_it = to_iterator(dataset)
#
#     xy_iter = chain.from_iterable(map(lambda xy: arr_func(xy), xy_it))
#
#     return from_iterator(xy_iter)


def on_x(dataset: Dataset, trs: Callable[[ndarray], ndarray])->Dataset:
    return Dataset.from_arrays(
        X=trs(dataset.X),
        Y=dataset.Y,
    )
