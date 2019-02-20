from numpy.core.multiarray import ndarray

from nautilus.data.dataset.array_dataset import ArrayDataset
from nautilus.data.dataset.dataset import Dataset
from nautilus.data.dataset.sample.array_sample_dataset import XYSampleDataset
from nautilus.data.dataset.sample_dataset import SampleDataset
from nautilus.data.dataset.train_test_dataset import TrainTestDataset
from nautilus.data.dataset.transform_dataset import TransformDataset
from nautilus.transform.transform import Transform


class DatasetUtils(object):
    """"""

    @staticmethod
    def from_array(array: ndarray)->ArrayDataset:
        return ArrayDataset(array)

    @staticmethod
    def from_xy_array(xarray: ndarray, yarray: ndarray):
        return XYSampleDataset(
            x_dataset=DatasetUtils.from_array(xarray),
            y_dataset=DatasetUtils.from_array(yarray),
        )

    @staticmethod
    def from_dataset_and_trs(dataset: SampleDataset, transform: Transform)->TransformDataset:
        return TransformDataset(
            dataset=dataset,
            transform=transform
        )

    @staticmethod
    def traintest_apply(traintest_dataset: TrainTestDataset, trs: Transform)->TrainTestDataset:
        return TrainTestDataset(
            train_dataset=DatasetUtils.from_dataset_and_trs(traintest_dataset.train, trs),
            test_dataset=DatasetUtils.from_dataset_and_trs(traintest_dataset.test, trs)
        )
