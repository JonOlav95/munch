import os
import json
import tensorflow as tf

from test import test
from train import train
from config import FLAGS


if __name__ == "__main__":

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["localhost:12345", "localhost:23456"]
        },
        'task': {'type': 'worker', 'index': 0}
    })

    #num_workers = 2
    #per worker batch size
    #global/total batch size

    if FLAGS["debug"]:
        tf.config.run_functions_eagerly(True)

    if FLAGS["train"]:
        train()
    elif FLAGS["test"]:
        test()
    else:
        print("No mode selected.")
