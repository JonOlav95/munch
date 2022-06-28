import matplotlib.pyplot as plt
import tensorflow as tf

from config import FLAGS
from mask import create_mask, reiterate_mask, mask_batch


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
    mask = create_mask(FLAGS["img_size"][:2])
    masked_batch = mask_batch(batch, mask)
    masks = reiterate_mask(mask)

    generated_image = generator([masked_batch, masks], training=False)

    stage1_gen = generated_image[0]
    stage2_gen = generated_image[1]

    stage1_gen = stage1_gen[0, ...]
    stage2_gen = stage2_gen[0, ...]

    x = masked_batch[0, ...]
    y = batch[0, ...]

    #disc_gen_result = discriminator([generated_image], training=False)
    #disc_gen_result = round(tf.math.reduce_mean(disc_gen_result).numpy(), 2)

    images = [x, stage1_gen, stage2_gen, y]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.imshow(images[i], cmap="gray")

    plt.show()