from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from nautilus.model.pipeline_model import PipelineModel
from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.sequential import Sequential
from nautilus.transform.tensor.flatten import Flatten
from nautilus.transform.tensor.resize import Resize

pca = PCA(n_components=8)
nn = KNeighborsClassifier()
pipeline = Pipeline(steps=[('pca', pca), ('nn', nn)])
PCA_NN = PipelineModel(SkModel.from_sklearn(pipeline),
                               in_trs=Sequential.from_transforms(Resize(target_size=(16, 16)), Flatten()), )
