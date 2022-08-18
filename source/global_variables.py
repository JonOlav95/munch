import tensorflow as tf
from config import FLAGS
from data_handler import load_data
from discriminator_gated import discriminator_gated
from discriminator_patchgan import discriminator_patchgan
from generator_gated import gated_generator
from generator_standard import generator_standard

generator = None
discriminator = None
ds = load_data()
generator_optimizer = None
discriminator_optimizer = None
checkpoint = None


def init_variables():
    global generator
    global discriminator
    global ds
    global generator_optimizer
    global discriminator_optimizer
    global checkpoint

    generator = generator_standard(FLAGS.get("img_size"))
    discriminator = discriminator_patchgan(FLAGS.get("img_size"))

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


if FLAGS["num_gpus"] > 1:

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        init_variables()

        # DS outside strategy?
        ds = strategy.experimental_distribute_dataset(ds)

else:
    init_variables()
