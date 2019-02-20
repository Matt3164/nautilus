from nautilus.context.prediction_context import BufferPredictionContext
from nautilus.context.train_context import TrainContext
from nautilus.data.dataset.sample.array_sample_dataset import XYSampleDataset
from nautilus.transform.dataset.to_tensor import DatasetToTensor
from nautilus.transform.transform import Transform


class ToPredictionContext(Transform):
    """"""

    def __call__(self, train_ctx: TrainContext)-> BufferPredictionContext:
        assert isinstance(train_ctx.dataset.train, XYSampleDataset)

        Xtrain, Ytrain = train_ctx.dataset.train.x_data.array, train_ctx.dataset.train.y_data.array

        Ytrain = Ytrain.reshape(-1, 1)

        Ytrain_pred = train_ctx.model.model.estimator.predict_proba(Xtrain)

        Xtest, Ytest = train_ctx.dataset.test.x_data.array, train_ctx.dataset.test.y_data.array

        Ytest = Ytest.reshape(-1, 1)

        Ytest_pred = train_ctx.model.model.estimator.predict_proba(Xtest)

        return BufferPredictionContext.from_arrays(
            Ytrain_pred.argmax(axis=1),
            Ytrain,
            Ytest_pred.argmax(axis=1),
            Ytest
        )


