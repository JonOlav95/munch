import os
import tensorflow as tf
import numpy as np
from PIL import Image
from config import FLAGS
from mask import mask_image

import matplotlib.pyplot as plt


def normalize(img):
    img = img / 255.
    img = (img * 2) - 1
    return img


def load_data(size=FLAGS["training_samples"]):
    """Custom function used to load unlabeled data from a folder.

    If the folder contains subfolders, the function explores those subfolders and returns
    the elements within. If the number of samples is less than the size argument,
    all files in the folders is loaded. Function is compatiable with both grayscale
    and RGB images.

    Args:
        size: The size of the dataset as an integer.

    Returns:
        The dataset as a Tensor split into three objects; the groundtruth, the maksed image,
        and the mask. Shape of the dataset is (batch_size, 3, width, heigh, channels).
    """
    path = FLAGS["dataset_dir"]
    _, dirs, filenames = next(os.walk(path))

    # Get all filenames in the folder(s)
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

        if channels == 1:
            groundtruth = np.expand_dims(groundtruth, axis=2)

        # Create the mask and a masked image.
        masked_img, mask = mask_image(groundtruth)

        # Normalize the data to a range between -1 and 1
        groundtruth = normalize(groundtruth)
        masked_img = normalize(masked_img)
        mask = normalize(mask)

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
