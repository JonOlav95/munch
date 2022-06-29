import tensorflow as tf

from data_handler import load_data
from patch_discriminator import discriminator
from gated_generator import gated_generator
from config import FLAGS, generator_optimizer, discriminator_optimizer
from plotter import plot_all


def test():
    model = gated_generator(FLAGS.get("img_size"))
    disc = discriminator(FLAGS.get("img_size"))
    ds = load_data(FLAGS["testing_samples"])

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=model,
                                     discriminator=disc)

    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    plot_all(ds, disc, model)
