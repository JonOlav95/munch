import yaml
import tensorflow as tf

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

f = open("parameters.yml", "r")
FLAGS = yaml.safe_load(f)
f.close()
