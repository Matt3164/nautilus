from typing import Callable, Iterator

from numpy.core.multiarray import ndarray
from nautilus.model.model import Model
import numpy as np

from nautilus.transform.identity import Identity


def iterator_on_array(array: ndarray)->Iterator[ndarray]:
    for i in range(array.shape[0]):
        yield array[i,::]

class PipelineModel(Model):
    """"""

    def __init__(self,
                 model: Model,
                 in_trs: Callable[[ndarray], ndarray]=Identity(),
                 out_trs: Callable[[ndarray], ndarray]=Identity(),
                 ):
        """Constructor for PipelineModel"""
        self.in_transform = in_trs
        self.out_transform = out_trs
        self.model=model

    def features(self, batch: ndarray)->ndarray:
        features = map(self.in_transform, iterator_on_array(batch))
        return np.array(list(features))

    def predict_on_batch(self, batch: ndarray)->ndarray:
        predictions = self.model(self.features(batch))
        out = map(self.out_transform, iterator_on_array(predictions))
        return np.array(list(out))

    def __call__(self, tensor: ndarray) -> ndarray:
        return self.predict_on_batch(tensor)

    @property
    def features_trs(self)->Callable[[ndarray], ndarray]:
        return self.in_transform





