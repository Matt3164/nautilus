from nautilus.transform.sample_transform import SampleTransform
from nautilus.transform.transform import Transform


class SampleTransformUtils(object):
    """"""

    @staticmethod
    def on_x(trs: Transform)->SampleTransform:
        return SampleTransform(x_transform=trs)

    @staticmethod
    def on_y(trs: Transform) -> SampleTransform:
        return SampleTransform(x_transform=trs)

SampleTrsUtils=SampleTransformUtils
