
from nautilus.data.dataset.sample_dataset import SampleDataset
from nautilus.data.dataset.tensor_dataset import TensorDataset
from nautilus.data.sample.sample import Sample


class XYSampleDataset(SampleDataset):
    """"""

    def __init__(self,
                 x_dataset: TensorDataset,
                 y_dataset: TensorDataset
                 ):
        """Constructor for ArraySampleDataset"""
        self.x_data=x_dataset
        self.y_data=y_dataset

    def __len__(self):
        assert len(self.x_data)==len(self.y_data)
        return len(self.x_data)

    def __getitem__(self, item) -> Sample:
        return Sample(
            x=self.x_data[item],
            y=self.y_data[item],
        )




