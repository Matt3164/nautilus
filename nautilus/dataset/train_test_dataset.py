from nautilus.dataset.dataset import Dataset


class TrainTestDataset(object):
    """"""

    def __init__(self,
                 train_dataset: Dataset,
                 test_dataset: Dataset):
        """Constructor for TrainTestDataset"""
        self.train=train_dataset
        self.test=test_dataset


