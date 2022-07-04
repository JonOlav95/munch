import matplotlib.pyplot as plt
import tensorflow as tf

from config import FLAGS
from mask import mask_image_batch


def plot_all(ds, discriminator, generator):
    """Plot an image for each batch"""
    for batch in ds:
        inner_plot(batch, discriminator, generator)


def plot_one(ds, discriminator, generator):
    """Plot one image from the dataset"""
    ds = ds.shuffle(16)
    for batch in ds.take(1):
        inner_plot(batch, discriminator, generator)


def inner_plot(batch, discriminator, generator):

    if len(batch) != FLAGS["global_batch_size"]:
        return

    masked_batch, masks = mask_image_batch(batch, n=len(batch))

    generated_image = generator([masked_batch, masks], training=False)

    stage1_gen = generated_image[0]
    stage2_gen = generated_image[1]

    for i in range(len(stage2_gen)):

        s1_img = stage1_gen[i, ...].numpy()
        s2_img = stage2_gen[i, ...].numpy()

        x = masked_batch[i, ...]
        y = batch[i, ...]

        #disc_gen_result = discriminator([generated_image], training=False)
        #disc_gen_result = round(tf.math.reduce_mean(disc_gen_result).numpy(), 2)

        images = [x, s1_img, s2_img, y]

        fig = plt.figure()
        for i in range(len(images)):
            fig.add_subplot(1, len(images), i + 1)
            plt.axis("off")
            plt.imshow(images[i], cmap="gray")

        plt.show()