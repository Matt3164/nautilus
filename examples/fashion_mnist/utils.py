from sklearn.externals import joblib

from nautilus.train.sklearn.sk_buffer_trainer import SkBufferTrainer
from nautilus.transform.context.to_array_context import ToArrayContext
from nautilus.transform.context.to_prediction_context import ToPredictionContext

mem = joblib.Memory("cache")


@mem.cache
def cached_array_context(ctx):
    return ToArrayContext()(ctx)

@mem.cache
def cached_train(ctx):
    return SkBufferTrainer()(ctx)

@mem.cache
def cached_pred_ctx(ctx):
    return ToPredictionContext()(ctx)