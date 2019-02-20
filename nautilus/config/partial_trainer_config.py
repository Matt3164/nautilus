class PartialTrainerConfig(object):
    """"""

    def __init__(self,
                 bath_size=32,
                 n_epoch=3,
                 classes=range(10)):
        """Constructor for PartialTrainerConfig"""
        self.batch_size=bath_size
        self.n_epoch=n_epoch
        self.classes=classes
