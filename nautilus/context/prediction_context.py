from numpy.core.multiarray import ndarray


class BufferPredictionCtx(object):
    """"""

    def __init__(
            self,
            y_true: ndarray,
            y_pred: ndarray
    ):
        """Constructor for BufferYTrueYPred"""
        self.y_true = y_true
        self.y_pred = y_pred