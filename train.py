import tensorflow as tf
import statistics
from model import *
from main import *


def loss_function(output, y):

    y = tf.expand_dims(y, axis=3)
    l1_abs = tf.abs(y - output)
    l1_loss = tf.reduce_mean(l1_abs)

    return l1_loss


@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        output = model(x)
        loss = loss_function(output, y)

    gradients = tape.gradient(loss, model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train(model, disc, ds):
    epochs = FLAGS["max_iters"]

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=model,
                                     discriminator=discriminator)

    for i in range(epochs):

        loss_arr = []

        for batch in ds:
            x = batch[..., :2]
            y = batch[..., 2]
            loss = train_step(model, x, y)

            loss_arr.append(loss.numpy())

        if i % 100 == 0:
            checkpoint.save()

        print("loss: " + str(statistics.mean(loss_arr)))
