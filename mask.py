import itertools

import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf

from config import FLAGS


def create_circular_mask(w, h):
    max_radius = w / 2
    radius = random.randrange(int(max_radius * 0.3), int(max_radius * 0.6))
    center = (random.randrange(radius, h - radius), random.randrange(radius, w - radius))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def create_rec_mask(w, h):
    rec_mask = np.full((w, h), False)

    rec_size = random.randrange(int(w * 0.2), int(w * 0.6))

    rec_x1 = random.randrange(0, w - rec_size)
    rec_y1 = random.randrange(0, h - rec_size)

    rec_x2 = rec_x1 + rec_size
    rec_y2 = rec_y1 + rec_size

    rec_mask[rec_x1:rec_x2, rec_y1:rec_y2] = True

    return rec_mask


def create_mask(dim):
    w = dim[0]
    h = dim[1]

    circle_mask = create_circular_mask(w, h)
    rec_mask = create_rec_mask(w, h)

    mask = np.logical_or(circle_mask, rec_mask)
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=2)

    return mask


def merge_mask_img(img, mask):
    img[mask[:] == 1] = 1
    return img


def merge_mask(img, mask):
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j][0] == 1:
                img[i][j] = np.array([0, 0, 0])

    return img


def mask_batch(groundtruth_batch, mask):
    groundtruth_cast = groundtruth_batch.numpy()

    arr = []
    for image in groundtruth_cast:
        arr.append(np.where(mask == 0, image, mask))

    return tf.convert_to_tensor(arr)


def reiterate_mask(mask, batch_size=FLAGS["batch_size"]):
    mask = np.expand_dims(mask, axis=0)
    masks = np.repeat(mask, batch_size, axis=0)
    return tf.convert_to_tensor(masks)
