import os

from sklearn.externals import joblib

from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.transform import Transform


class SkModelToPath(Transform):
    """"""

    def __init__(self,
                 path: str):
        """Constructor for SkModelToPath"""
        self.path = path

    def __call__(self, data: SkModel)->str:

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        joblib.dump(
            data,
            self.path
        )
        return self.path




