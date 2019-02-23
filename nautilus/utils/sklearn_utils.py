from sklearn.base import BaseEstimator
from sklearn.externals import joblib

from nautilus.utils import file_utils


def model_from_path(path: str)->BaseEstimator:
    assert file_utils.exists(path)
    return joblib.load(path)


def model_to_path(model: BaseEstimator, path:str)->str:

    file_utils.mk_parent(path)

    joblib.dump(
        model,
        path
    )

    assert file_utils.exists(path)
    return path




