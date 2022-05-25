import os
import tensorflow as tf
import numpy as np

from PIL import Image
from mask import create_mask, merge_mask_img
from main import FLAGS


def load_data():
    path = FLAGS["dataset_dir"]
    _, _, filenames = next(os.walk(path))

    filenames = np.array(filenames)

    ds_size = len(filenames)
    if ds_size > FLAGS["training_samples"]:
        ds_size = FLAGS["training_samples"]

    ds = np.empty(ds_size, dtype=object)

    img_size = FLAGS["img_size"][:2]

    for i in range(ds_size):
        groundtruth = Image.open(path + filenames[i]).convert("L")
        groundtruth = groundtruth.resize(img_size)

        groundtruth = np.array(groundtruth, dtype="float32")
        groundtruth = groundtruth / 255.

        mask = create_mask(img_size)
        masked_img = merge_mask_img(np.copy(groundtruth), mask)

        groundtruth = np.expand_dims(groundtruth, axis=2)
        masked_img = np.expand_dims(masked_img, axis=2)
        mask = np.expand_dims(mask, axis=2)

        masked_img = tf.convert_to_tensor(masked_img)
        mask = tf.convert_to_tensor(mask)
        groundtruth = tf.convert_to_tensor(groundtruth)

        data = tf.concat((masked_img, mask, groundtruth), axis=2)

        ds[i] = data

    ds = tf.data.Dataset.from_tensor_slices(ds.tolist())
    ds = ds.batch(FLAGS["batch_size"])

    return ds
