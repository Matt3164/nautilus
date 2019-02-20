from nautilus.transform.transform import Transform


class Identity(Transform):
    """"""

    def __call__(self,data):
        return data


