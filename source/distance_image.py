from scipy import ndimage
from data_handler import load_data
import matplotlib.pyplot as plt
import numpy as np


ds = load_data(100)
ds = ds.shuffle(buffer_size=10)

for batch in ds.take(1):

    img = batch[0]

    img = img.numpy()

    threshold = 0.7
    binary_img = 1.0 * (img > threshold)

    invert_img = 1 - binary_img
    distance_img = ndimage.distance_transform_edt(binary_img)

    images = [img, binary_img, invert_img, distance_img]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.imshow(images[i], cmap="gray")

    plt.show()
