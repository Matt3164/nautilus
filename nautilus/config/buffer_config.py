from nautilus.config.train_config import TrainingConfig


class BufferConfig(TrainingConfig):
    """"""

    def __init__(self,
                 subsample: float=None
                 ):
        """Constructor for SkConfig"""
        super(BufferConfig, self).__init__()
        self.subsample = subsample
