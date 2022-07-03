import io
import json
import base64
import logging
import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, request, jsonify, abort
from config import FLAGS
from generator_gated import gated_generator
from patch_discriminator import discriminator

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

generator = gated_generator(FLAGS.get("img_size"))
disc = discriminator(FLAGS.get("img_size"))
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=disc)

checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))


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
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)
    #gen_output = generator

    # process your img_arr here
    with open("../bird2.png", "rb") as f:
        im_bytes = f.read()

    return_img = base64.b64encode(im_bytes).decode("utf8")
    payload = json.dumps({"image": return_img})

    return jsonify({
        'img': return_img
    })


def run_server_api():
    app.run(host='0.0.0.0', port=8080)


if __name__ == "__main__":
    run_server_api()
