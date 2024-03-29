import matplotlib.pyplot as plt
import tensorflow as tf

from config import FLAGS
from datetime import datetime


def plot_all(ds, generator):
    """Plot an image for each batch"""
    for batch in ds:
        inner_plot(batch, generator)


def plot_one(ds, generator):
    """Plot one image from the dataset"""
    ds = ds.shuffle(16)
    for batch in ds.take(1):
        inner_plot(batch, generator)


def inner_plot(batch, generator):

    if len(batch) != FLAGS["global_batch_size"]:
        return

    gr_batch = batch[0, 0, ...]
    masked_batch = batch[0, 1, ...]
    mask_batch = batch[0, 2, ...]

    generated_image = generator([masked_batch, mask_batch], training=False)

    generated_image_1 = generated_image[0]
    generated_image_2 = generated_image[1]

    for i in range(len(generated_image_1)):

        gen_img_1 = generated_image_1[i, ...].numpy()
        gen_img_2 = generated_image_2[i, ...].numpy()

        x = masked_batch[i, ...]
        y = gr_batch[i, ...]

        images = [x, gen_img_1, gen_img_2, y]

        fig = plt.figure()
        for j in range(len(images)):
            fig.add_subplot(1, len(images), j + 1)
            plt.axis("off")
            plt.imshow(images[j] * 0.5 + 0.5, cmap="gray")

        # Can be used to store the plotted image.
        #fig.savefig(fname=FLAGS["plot_dir"] + str(datetime.now().strftime("%Y%m%d%H%M%S")), format="png")
        plt.show()
