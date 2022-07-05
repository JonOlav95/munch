import tensorflow as tf

from data_handler import load_data
from generator_standard import generator_standard
from patch_discriminator import discriminator
from generator_gated import gated_generator
from config import FLAGS
from plotter import plot_all


def test():
    generator = gated_generator(FLAGS.get("img_size"))
    disc = discriminator(FLAGS.get("img_size"))

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=disc)

    ds = load_data(FLAGS["testing_samples"])

    checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    plot_all(ds, generator)

