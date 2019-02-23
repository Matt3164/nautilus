from typing import List

from nautilus.data.sample.sample import Sample

from nautilus.imatools.features import OrbDescriptor
from nautilus.transform.transform import Transform


class OrbFlatMap(Transform):
    """"""

    def __init__(self,
                 n_keypoints:int=100):
        """Constructor for OrbDescriptors"""
        self.n_keypoints=n_keypoints

    def __call__(self, data: Sample)->List[Sample]:

        descriptors = OrbDescriptor(n_keypoints=self.n_keypoints)(data.x)

        return list(
            map( lambda i: Sample(x=descriptors[i,::], y=data.y), range(descriptors.shape[0]))
        )


