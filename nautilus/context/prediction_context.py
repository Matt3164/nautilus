from numpy.core.multiarray import ndarray


class BufferPredictionCtx(object):
    """"""

    def __init__(
            self,
            y_true: ndarray,
            y_pred: ndarray,
            y_pred_prob: ndarray
    ):
        """Constructor for BufferYTrueYPred"""
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob