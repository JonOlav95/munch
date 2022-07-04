import matplotlib.pyplot as plt
import tensorflow as tf

from config import FLAGS
from mask import mask_image_batch


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

    gr_batch = batch[:, 0, ...]
    masked_batch = batch[:, 1, ...]

    generated_image = generator(masked_batch, training=False)

    generated_image_1 = generated_image[0]
    generated_image_2 = generated_image[1]
    generated_image_3 = generated_image[2]

    for i in range(len(generated_image_1)):

        gen_img_1 = generated_image_1[i, ...].numpy()
        gen_img_2 = generated_image_2[i, ...].numpy()
        gen_img_3 = generated_image_3[i, ...].numpy()

        x = masked_batch[i, ...]
        y = gr_batch[i, ...]

        #disc_gen_result = discriminator([generated_image], training=False)
        #disc_gen_result = round(tf.math.reduce_mean(disc_gen_result).numpy(), 2)

        images = [x, gen_img_1, gen_img_2, gen_img_3, y]

        fig = plt.figure()
        for i in range(len(images)):
            fig.add_subplot(1, len(images), i + 1)
            plt.axis("off")
            plt.imshow(images[i], cmap="gray")

        plt.show()
