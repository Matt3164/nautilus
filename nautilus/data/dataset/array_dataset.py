from numpy.core.multiarray import ndarray

from nautilus.data.dataset.tensor_dataset import TensorDataset


class ArrayDataset(TensorDataset):
    """"""
    
    def __init__(self,array: ndarray):
        """Constructor for ArrayDataset"""
        super(ArrayDataset, self).__init__()
        self.array=array

    def __getitem__(self, item):
        return self.array[item,::]

    def __len__(self):
        return self.array.shape[0]


        