import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
from config import FLAGS
from mask import create_mask


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
    _, dirs, filenames = next(os.walk(path))

    if dirs:
        filenames = []
        for d in dirs:
            _, _, f = next(os.walk(path + d))
            filenames.extend([d + "/" + x for x in f])
    else:
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
        groundtruth = (groundtruth * 2) - 1

        if channels == 1:
            groundtruth = np.expand_dims(groundtruth, axis=2)

        dim = FLAGS["img_size"][:2]
        mask = create_mask(dim)

        masked_img = np.where(mask == 0, groundtruth, mask)

        if channels == 3:
            squeeze_mask = np.squeeze(mask)
            mask = np.zeros(FLAGS["img_size"], dtype="float32")
            mask[:, :, 0] = squeeze_mask
            mask[:, :, 1] = squeeze_mask
            mask[:, :, 2] = squeeze_mask

        groundtruth = tf.convert_to_tensor(groundtruth)
        masked_img = tf.convert_to_tensor(masked_img)
        mask = tf.convert_to_tensor(mask)

        ds[i] = [groundtruth, masked_img, mask]

    ds = tf.data.Dataset.from_tensor_slices(ds.tolist())
    ds = ds.batch(FLAGS["global_batch_size"])

    return ds
