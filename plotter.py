import matplotlib.pyplot as plt
import tensorflow as tf


def plot_all(ds, discriminator, generator):
    for batch in ds:
        inner_plot(batch, discriminator, generator)


def plot_one(ds, discriminator, generator):
    ds = ds.shuffle(16)
    for batch in ds.take(1):
        inner_plot(batch, discriminator, generator)


def inner_plot(batch, discriminator, generator):
    x, mask, y = tf.split(value=batch, num_or_size_splits=3, axis=3)

    generated_image = generator([x, mask], training=False)
    generated_image = generated_image[0, ...]

    x = x[0, ...]
    mask = mask[0, ...]
    y = y[0, ...]

    #disc_gen_result = discriminator([generated_image], training=False)
    #disc_gen_result = round(tf.math.reduce_mean(disc_gen_result).numpy(), 2)

    images = [x, mask, generated_image, y]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.imshow(images[i], cmap="gray")

    plt.show()