import os
import tensorflow as tf
import numpy as np

from PIL import Image
from mask import create_mask, merge_mask_img, merge_mask
from config import FLAGS


def tmp_colored():
    pass

def tmp_grayscale():
    pass


def load_data(size):
    path = FLAGS["dataset_dir"]
    _, _, filenames = next(os.walk(path))

    filenames = np.array(filenames)

    ds_size = len(filenames)
    if ds_size > size:
        ds_size = size

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

        mask = create_mask(img_size)
        mask = np.expand_dims(mask, axis=2)

        if channels == 1:
            masked_img = merge_mask_img(np.copy(groundtruth), mask)
        else:
            masked_img = merge_mask(np.copy(groundtruth), mask)

        groundtruth = np.expand_dims(groundtruth, axis=3)
        masked_img = np.expand_dims(masked_img, axis=3)
        mask = np.expand_dims(mask, axis=3)

        masked_img = tf.convert_to_tensor(masked_img)
        mask = tf.convert_to_tensor(mask)
        groundtruth = tf.convert_to_tensor(groundtruth)

        data = tf.concat((masked_img, mask, groundtruth), axis=3)

        ds[i] = data

    ds = tf.data.Dataset.from_tensor_slices(ds.tolist())
    ds = ds.batch(FLAGS["batch_size"])

    return ds
