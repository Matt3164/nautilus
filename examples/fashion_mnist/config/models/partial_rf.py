from nautilus.model.pipeline_model import PipelineModel
from nautilus.model.sklearn.sklearn_model import SkModel
from sklearn.ensemble import RandomForestClassifier

from nautilus.imatools.features.haar import Haar
from nautilus.transform.sequential import Sequential
from nautilus.transform.tensor.resize import Resize

PARTIAL_RF = PipelineModel(SkModel.from_partial_sklearn_ensemble(RandomForestClassifier(max_depth=3)),
                           in_trs=Sequential.from_transforms(Resize(target_size=(8, 8)), Haar()), )
