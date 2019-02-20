from numpy.core.records import ndarray


class Sample(object):
    """"""

    def __init__(self,
                 x: ndarray,
                 y: ndarray
                 ):
        """Constructor for Sample"""
        self.x = x
        self.y = y
