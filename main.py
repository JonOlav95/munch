import yaml
import tensorflow as tf
from train import *

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

f = open("parameters.yml", "r")
FLAGS = yaml.safe_load(f)
f.close()


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    train()
