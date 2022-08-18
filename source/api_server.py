import io
import json
import base64
import logging
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from skimage.filters import threshold_otsu, threshold_local
from PIL import Image
from flask import Flask, request, jsonify, abort
from config import FLAGS
from discriminator_gated import discriminator_gated
from generator_gated import gated_generator

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

generator = gated_generator(FLAGS["img_size"])
discriminator = discriminator_gated(FLAGS["img_size"])
g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                 discriminator_optimizer=discriminator,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))


def grayscale_to_binary(img):
    global_thresh = threshold_otsu(img)
    binary_global = img > global_thresh

    block_size = 35
    local_thresh = threshold_local(img, block_size, offset=10)
    binary_local = img > local_thresh

    return binary_local


def generate_image(numpy_image):

    mask = np.where(numpy_image >= 254, 255, 0)


    plt.imshow(mask, cmap="gray")
    plt.show()


    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=-1)

    mask = mask / 255.
    mask = (mask * 2) - 1

    img = np.expand_dims(numpy_image, axis=0)
    img = np.expand_dims(img, axis=-1)

    img = img / 255.
    img = (img * 2) - 1

    img_tensor = tf.convert_to_tensor(img)
    mask_tensor = tf.convert_to_tensor(mask)

    gen_output = generator([img_tensor, mask_tensor], training=False)
    gen_img = gen_output[0]
    gen_img = gen_img.numpy()

    gen_img = (gen_img * 0.5) + 0.5
    gen_img = np.squeeze(gen_img)

    gen_img *= 255.

    plt.imshow(gen_img, cmap="gray")
    plt.show()

    return gen_img


@app.route("/test", methods=['POST'])
def test_method():
    # print(request.json)
    if not request.json or 'image' not in request.json:
        abort(400)

    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # PIL image object to numpy array
    img_arr = np.asarray(img)
    img_arr = cv2.resize(img_arr, dsize=(256, 256))

    plt.imshow(img_arr, cmap="gray")
    plt.show()

    #binary_image = grayscale_to_binary(img_arr)

    # plt.imshow(binary_image, cmap="gray")
    # plt.show()

    gen_img = generate_image(img_arr)

    # plt.imshow(gen_img, cmap="gray")
    # plt.show()

    # gen_img = cv2.resize(gen_img, dsize=(3508, 2480))

    _, enc = cv2.imencode(".png", gen_img)
    content = enc.tobytes()

    content = base64.b64encode(content).decode("utf8")

    return jsonify({
        'img': content
    })


def run_server_api():
    app.run(host='0.0.0.0', port=8080)


if __name__ == "__main__":
    run_server_api()