import requests
import PIL.Image as pilimg
import json
import numpy as np
import os
import argparse

import matplotlib.pyplot as plt

# # Test
# img_path = os.path.join(os.getcwd(), "dataset", "abnormal", "1.jpg")
# img = pilimg.open(img_path)

# plt.imshow(img)
# plt.show()

# Argument set
parser = argparse.ArgumentParser(
    description="Qualcomm pair space check application")
parser.add_argument(
    "--img", dest="img_path", required=True, help="Classfication image file path"
)

args = parser.parse_args()

# Exist check image file
if not os.path.isfile(args.img_path):
    print("Image file not found... [{}]".format(args.img_path))

# Image load
img = pilimg.open(args.img_path)

# Image to json
img = np.array(img)
img = img[np.newaxis, ...]
# print('shape:', img.shape)

data = json.dumps({"signature_name": "serving_default",
                  "instances": img.tolist()})
# print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

headers = {"content-type": "application/json"}
json_response = requests.post(
    "http://localhost:8501/v1/models/qualcomm_pair_space_check:predict",
    data=data,
    headers=headers,
)
predictions = json.loads(json_response.text)["predictions"]
predictions = np.array(predictions).flatten()
print(predictions)
predictions = np.where(predictions < 0.5, 0, 1)

print(predictions[0])
