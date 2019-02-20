from nautilus.config.train_config import TrainingConfig
from nautilus.data.dataset.dataset import Dataset
from nautilus.data.dataset.train_test_dataset import TrainTestDataset
from nautilus.model.model import Model


class TrainContext(object):
    """"""

    def __init__(self,
                 dataset: TrainTestDataset,
                 model: Model,
                 config: TrainingConfig):
        """Constructor for TrainContext"""
        self.model = model
        self.config = config
        self.dataset = dataset
