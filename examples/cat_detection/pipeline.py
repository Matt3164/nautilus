from glob import glob
from random import randint

import cv2
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from numpy import zeros_like, clip, concatenate, expand_dims
from numpy.ma import array
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt

from examples.cat_detection.data.cat_loader import CatLoader

if __name__ == '__main__':

    data_path = "/home/matthieu/Workspace/data/cats"

    data = CatLoader(root_path=data_path, folders=["CAT_00"])

    sample = data[0]
    print(sample.x.shape)
    print(sample.y.shape)

    z = zeros_like(sample.x[:,:,0])
    coords = clip(sample.y, a_min=0, a_max=None)
    z[coords[2]:coords[3], coords[0]:coords[1]]=255

    plt.subplot(1,2,1)
    plt.imshow(sample.x)
    plt.subplot(1, 2, 2)
    plt.imshow(z)
    plt.show()

    b = concatenate([sample.x, expand_dims(z, axis=-1)], axis=-1)

    print(extract_patches_2d(b, (256, 256), max_patches=100).shape)








    # print( extract_patches_2d(cv2.imread(list_images[0]), (256, 256), max_patches=100).shape )
    #
    # idx = randint(0, len(list_images))
    #
    # with open(list_images[idx]+".cat", "r") as f:
    #     content = f.read()
    #
    # print(content)
    #
    # pos = [ int(e) for e in content.split(" ") if e!='']
    #
    # print(pos)
    #
    # pos = array(pos)[1:]
    #
    # x = pos[::2]
    #
    # print(x)
    #
    # y = pos[1::2]
    #
    # print(y)
    #
    # xmin, xmax = x.min(), x.max()
    # ymin, ymax = y.min(), y.max()
    #
    # xmin = max(xmin, 0)
    # ymin = max(ymin, 0)
    #
    # im = cv2.imread(list_images[idx])
    #
    # im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=5)
    #

    #
    # Bbox.from_bounds()


