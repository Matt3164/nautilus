from sklearn.externals import joblib

from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.transform import Transform


class PathToSkModel(Transform):
    def __call__(self, data: str)->SkModel:
        return joblib.load(data)