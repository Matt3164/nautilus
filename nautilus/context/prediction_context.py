from numpy.core.multiarray import ndarray


class BufferYTrueYPred(object):
    """"""

    def __init__(
            self,
            y_true: ndarray,
            y_pred: ndarray
    ):
        """Constructor for BufferYTrueYPred"""
        self.y_true = y_true
        self.y_pred = y_pred


class BufferPredictionContext(object):
    """"""

    def __init__(self,
                 train_truepred: BufferYTrueYPred,
                 test_truepred: BufferYTrueYPred
                 ):
        """Constructor for PredictionContext"""
        self.train = train_truepred
        self.test = test_truepred

    @staticmethod
    def from_arrays(y_train_pred: ndarray, y_train_true: ndarray, y_test_pred: ndarray, y_test_true: ndarray):
        return BufferPredictionContext(
            train_truepred=BufferYTrueYPred(
                y_true=y_train_true,
                y_pred=y_train_pred
            ),
            test_truepred=BufferYTrueYPred(
                y_true=y_test_true,
                y_pred=y_test_pred
            )
        )


