from nautilus.context.train_context import TrainContext
from nautilus.transform.transform import Transform

class Trainer(Transform):
    """"""

    def __call__(self, train_ctx: TrainContext)->TrainContext:
        raise NotImplementedError

    
