from typing import List

from nautilus.transform.transform import Transform


class Sequential(Transform):
    """"""

    def __init__(self,
                 transforms: List[Transform]):
        """Constructor for Sequentiel"""
        self.transforms = transforms

    def __call__(self, data):

        for trs in self.transforms:
            data = trs(data)

        return data

    @staticmethod
    def from_transforms(*iterable):
        return Sequential(
            list(iterable)
        )


