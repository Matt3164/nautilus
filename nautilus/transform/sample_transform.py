from nautilus.data.sample.sample import Sample
from nautilus.transform.identity import Identity
from nautilus.transform.transform import Transform


class SampleTransform(Transform):
    """"""

    def __init__(self,
                 x_transform: Transform=Identity(),
                 y_transform: Transform=Identity()):
        """Constructor for SampleTransform"""
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __call__(self, sample: Sample)->Sample:
        return Sample(
            x=self.x_transform(sample.x),
            y=self.y_transform(sample.y)
        )



