import os
import tensorflow as tf
import numpy as np

from PIL import Image
from config import FLAGS


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

        groundtruth = tf.convert_to_tensor(groundtruth)

        ds[i] = groundtruth

    ds = tf.data.Dataset.from_tensor_slices(ds.tolist())
    ds = ds.batch(FLAGS["batch_size"])

    return ds
