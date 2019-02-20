from nautilus.data.sample.sample import Sample
from nautilus.model.model import Model
from nautilus.transform.sample.classiftiler import ClassifTiler
from nautilus.transform.transform import Transform
import numpy as np

class BagOfWords(Transform):
    def __init__(self,
                 model: Model,
                 tiler: ClassifTiler,
                 n_max_features: int=2500
                 ):
        self.n_max_features = 2500
        self.model = model
        self.tiler = tiler

    def __call__(self, data: np.ndarray)->np.ndarray:

        fake_sample = Sample(x=data, y=0)

        extracted_samples = self.tiler(fake_sample)

        # Construc array from sample
        X = np.array(list(map(lambda s: s.x, extracted_samples)))

        # Use model to predict
        Y_pred = self.model.predict_on_batch(X)

        # Count predictions
        words, counts = np.unique(Y_pred, return_counts=True)
        # Create feature vector
        features = np.zeros((self.n_max_features,))
        features[words] = counts

        return features