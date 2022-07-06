import tensorflow as tf
from config import FLAGS
from data_handler import load_data
from discriminator_gated import discriminator_gated
from discriminator_patchgan import discriminator_patchgan
from generator_gated import gated_generator
from generator_sketch_tensor import st_generator
from generator_standard import generator_standard

generator = None
discriminator = None
ds = load_data(FLAGS["training_samples"])
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

    if FLAGS["generator"] == "generator_gated":
        generator = gated_generator(FLAGS.get("img_size"))
    elif FLAGS["generator"] == "generator_cnn":
        generator = generator_standard(FLAGS.get("img_size"))
    elif FLAGS["generator"] == "generator_st":
        generator = st_generator(FLAGS.get("img_size"))

    if FLAGS["discriminator"] == "discriminator_gated":
        discriminator = discriminator_gated(FLAGS.get("img_size"))
    elif FLAGS["discriminator"] == "discriminator_patchgan":
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
