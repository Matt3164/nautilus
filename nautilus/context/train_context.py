from sklearn.base import BaseEstimator

from nautilus.config.train_config import TrainingConfig
from nautilus.data.dataset.train_test_dataset import TrainTestDataset


class TrainContext(object):
    """"""

    def __init__(self,
                 dataset: TrainTestDataset,
                 model: BaseEstimator,
                 config: TrainingConfig):
        """Constructor for TrainContext"""
        self.model = model
        self.config = config
        self.dataset = dataset
