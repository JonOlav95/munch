import os
import tensorflow as tf
import numpy as np

from PIL import Image
from mask import create_mask, merge_mask_img
from main import FLAGS


def load_data(path="dataset/"):
    _, _, filenames = next(os.walk(path))

    filenames = np.array(filenames)

    ds_size = len(filenames)
    if ds_size > FLAGS["training_samples"]:
        ds_size = FLAGS["training_samples"]

    ds = np.empty(ds_size, dtype=object)

    for i in range(ds_size):
        img = Image.open(path + filenames[i]).convert("L")
        img = img.resize((256, 256))

        img = np.array(img, dtype="float32")
        img = img / 255.

        img_x = np.copy(img)
        mask = create_mask(256, 256)
        merge_mask_img(img, mask)

        img = np.expand_dims(img, axis=2)
        img_x = np.expand_dims(img_x, axis=2)
        mask = np.expand_dims(mask, axis=2)

        img_x = tf.convert_to_tensor(img_x)
        mask = tf.convert_to_tensor(mask)

        img = tf.convert_to_tensor(img)
        masked_img = tf.concat((img_x, mask, img), axis=2)

        ds[i] = masked_img

    ds = tf.data.Dataset.from_tensor_slices(ds.tolist())
    ds = ds.batch(2)

    return ds
