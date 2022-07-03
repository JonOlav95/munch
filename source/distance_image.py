from scipy import ndimage
import tensorflow as tf
from data_handler import load_data
import matplotlib.pyplot as plt
import numpy as np


def distance_to_img(distance_img):
    img = 1.0 * (distance_img > 0.95)

    return img
