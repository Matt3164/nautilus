from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from nautilus.model.pipeline_model import PipelineModel
from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.sequential import Sequential
from nautilus.transform.tensor.flatten import Flatten

kmeans = KMeans(n_clusters=100, n_init=1)
pca = PCA(n_components=8)
km_pipeline = Pipeline(steps=[('pca', pca), ('kmeans', kmeans)])
PCA_KMEANS = PipelineModel(SkModel.from_sklearn(km_pipeline), in_trs=Sequential.from_transforms(Flatten()), )

rf = RandomForestClassifier(max_depth=5, n_estimators=25)
pipeline = Pipeline(steps=[('normalizer', Normalizer()), ('rf', rf)])

BOW_MODEL = SkModel.from_sklearn(pipeline)
