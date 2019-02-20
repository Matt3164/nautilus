from sklearn.cluster import KMeans

from nautilus.model.pipeline_model import PipelineModel
from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.identity import Identity

kmeans = KMeans(n_clusters=100, n_init=1)
BOW_ORB_KM = PipelineModel(SkModel.from_sklearn(kmeans), in_trs=Identity(), )
