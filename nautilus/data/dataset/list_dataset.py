from nautilus.data.dataset.dataset import Dataset


class ListDataset(Dataset):
    """"""

    def __init__(self, elmts):
        """Constructor for ListDataset"""
        self.elmts = elmts

    def __len__(self):
        return len(self.elmts)

    def __getitem__(self, item):
        return self.elmts[item]
