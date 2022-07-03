import matplotlib.pyplot as plt
import tensorflow as tf

from config import FLAGS
from mask import mask_image_batch


def plot_one_distance(ds, discriminator, generator):
    """Plot one image from the dataset"""
    ds = ds.shuffle(4)
    for batch in ds.take(1):
        inner_plot_distance(batch, discriminator, generator)


def inner_plot_distance(batch, discriminator, generator):

    if len(batch) != FLAGS["global_batch_size"]:
        return

    distance_img_masked = batch[:, 0, ...]
    generated_image = generator(distance_img_masked, training=False)

    distance_img_masked = batch[0, 0, ...].numpy()
    gr_img_masked = 1 - 1.0 * (distance_img_masked == 0.)

    generated_image = generated_image[0].numpy()
    casted_generated_img = 1 - 1.0 * (generated_image < 0.01)

    distance_img_gr = batch[0, 1, ...].numpy()
    casted_gr_img = 1 - 1.0 * (distance_img_gr == 0.)

    images = [distance_img_masked, gr_img_masked, generated_image, casted_generated_img, distance_img_gr, casted_gr_img]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.imshow(images[i], cmap="gray")

    plt.show()
