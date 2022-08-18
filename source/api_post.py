import requests
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
import io

from PIL import Image

# api-endpoint
URL = "http://localhost:8080/test"

with open("../4994.JPEG", "rb") as f:
    im_bytes = f.read()

im_b64 = base64.b64encode(im_bytes).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
payload = json.dumps({"image": im_b64, "other_key": "value"})
response = requests.post(URL, data=payload, headers=headers)


img = response.json()["img"]
im_b64 = base64.b64decode(img.encode('utf-8'))

# convert bytes data to PIL Image object
img = Image.open(io.BytesIO(im_b64))

# PIL image object to numpy array
img_arr = np.asarray(img)

plt.imshow(img_arr)
plt.show()
