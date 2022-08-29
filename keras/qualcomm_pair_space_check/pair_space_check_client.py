import requests
import PIL.Image as pilimg
import json
import numpy as np
import os
import argparse

# Argument set
parser = argparse.ArgumentParser(
    description="Qualcomm pair space check application")
parser.add_argument(
    "--img", dest="img_path", required=True, help="Classfication image file path"
)
parser.add_argument(
    "--ip", dest="server_ip", default="localhost", help="Tensorflow serving server ip"
)
parser.add_argument(
    "--port", dest="server_port", default="8501", help="Tensorflow serving server port"
)

args = parser.parse_args()

# Exist check image file
if not os.path.isfile(args.img_path):
    print("Image file not found... [{}]".format(args.img_path))

try:
    # Image load
    img = pilimg.open(args.img_path)

    # Image to json
    img = np.array(img)
    img = img[np.newaxis, ...]

    data = json.dumps({"signature_name": "serving_default",
                      "instances": img.tolist()})

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        "http://"
        + args.server_ip
        + ":"
        + args.server_port
        + "/v1/models/qualcomm_pair_space_model:predict",
        data=data,
        headers=headers,
    )
    predictions = json.loads(json_response.text)["predictions"]
    predictions = np.array(predictions).flatten()
    pred = np.where(predictions < 0.5, 0, 1)

except:
    predictions = [-1]
    pred = [-1]

print("{} {:.2f}".format(pred[0], predictions[0]))
