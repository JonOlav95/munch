import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
from config import FLAGS


def distance_transform(img):
    threshold = 0.7
    binary_img = 1.0 * (img > threshold)
    distance_img = ndimage.distance_transform_edt(binary_img)

    images = [distance_img, binary_img]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.imshow(images[i], cmap="gray")

    plt.show()

    return binary_img, distance_img


def load_data(size):
    path = FLAGS["dataset_dir"]
    _, _, filenames = next(os.walk(path))

    filenames = np.array(filenames)

    ds_size = len(filenames)
    if ds_size > size:
        ds_size = size - 1

    ds = np.empty(ds_size, dtype=object)

    img_size = FLAGS["img_size"][:2]
    channels = FLAGS["img_size"][2]

    for i in range(ds_size):
        if channels == 1:
            groundtruth = Image.open(path + filenames[i]).convert("L")
        else:
            groundtruth = Image.open(path + filenames[i])

        groundtruth = groundtruth.resize(img_size)
        groundtruth = np.array(groundtruth, dtype="float32")
        groundtruth = groundtruth / 255.

        if channels == 1:
            groundtruth = np.expand_dims(groundtruth, axis=2)

        #binary_img, distance_img = distance_transform(groundtruth)
        groundtruth = tf.convert_to_tensor(groundtruth)
#
        #binary_img = tf.convert_to_tensor(binary_img)
        #distance_img = tf.convert_to_tensor(distance_img)

        ds[i] = groundtruth

    ds = tf.data.Dataset.from_tensor_slices(ds.tolist())
    ds = ds.batch(FLAGS["global_batch_size"])

    return ds
