import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def create_circular_mask(w, h):
    radius = random.randrange(50, 100)
    center = (random.randrange(radius, h - radius), random.randrange(radius, w - radius))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def create_rec_mask(w, h):
    rec_mask = np.full((w, h), False)

    rec_size = random.randrange(50, 100)

    rec_x1 = random.randrange(0, w - rec_size)
    rec_y1 = random.randrange(0, h - rec_size)

    rec_x2 = rec_x1 + rec_size
    rec_y2 = rec_y1 + rec_size

    rec_mask[rec_x1:rec_x2, rec_y1:rec_y2] = True

    return rec_mask


def create_mask(w, h):
    circle_mask = create_circular_mask(w, h)
    rec_mask = create_rec_mask(w, h)

    mask = np.logical_or(circle_mask, rec_mask)
    mask = mask.astype(np.float32)

    return mask


def merge_mask_img(img, mask):
    img[mask[:] == 1] = 1
    return img
