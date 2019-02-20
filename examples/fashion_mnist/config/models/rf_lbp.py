from sklearn.ensemble import RandomForestClassifier

from nautilus.imatools.transform.features.lbp import LBP
from nautilus.model.pipeline_model import PipelineModel
from nautilus.model.sklearn.sklearn_model import SkModel

RF_LBP = PipelineModel(SkModel.from_sklearn(RandomForestClassifier(max_depth=1, n_estimators=25)),
                               in_trs=LBP.from_radius(4), )