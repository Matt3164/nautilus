import os
from glob import glob
from itertools import chain
from typing import List

import cv2
from nautilus.data.sample.sample import Sample
from numpy.ma import array

from nautilus.dataset.dataset import Dataset


class CatLoader(Dataset):
    def __init__(self,
                 root_path: str,
                 folders: List[str],
                 max_images: int=10
                 ):
        self.root_path = root_path
        self.folders = folders
        self.max_images = max_images

        folders = list( map(lambda fold: os.path.join(self.root_path, fold, "*.jpg"), folders) )
        self.list_images = list(
            chain.from_iterable(
                map(lambda fold: glob(fold), folders)
            )
        )

        if self.max_images:
            self.list_images = self.list_images[:self.max_images]

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, item: int)->Sample:
        image_file = self.list_images[item]

        with open(image_file + ".cat", "r") as f:
            content = f.read()

        pos = [int(e) for e in content.split(" ") if e != '']

        pos = array(pos)[1:]

        x = pos[::2]

        y = pos[1::2]

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        return Sample(
            x=cv2.imread(image_file),
            y=array([xmin, xmax, ymin, ymax])
        )



