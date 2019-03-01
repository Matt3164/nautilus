from logging import warning
from typing import Tuple
from numpy.core.multiarray import ndarray

from nautilus.utils import file_utils, np_utils


class Dataset(object):
    """"""

    def __init__(self, X: ndarray, Y: ndarray):

        if not( X.shape[0]==Y.shape[0]):
            warning("Dataset X and Y do not have the same size")
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item)->Tuple[ndarray, ndarray]:
        return self.X[item, ::], self.Y[item, ::]

    @staticmethod
    def from_arrays(X: ndarray, Y: ndarray):
        return Dataset(X=X, Y=Y)

    @staticmethod
    def from_file(filepath: str):
        assert file_utils.exists(filepath)

        data = np_utils.load(filepath)

        return Dataset(
            X=data["X"],
            Y=data["Y"]
        )
    def to_file(self, filepath: str):
        file_utils.mk_parent(filepath)

        np_utils.save_npz(filepath, dict(X=self.X, Y=self.Y))

        return file_utils.exists(filepath)



