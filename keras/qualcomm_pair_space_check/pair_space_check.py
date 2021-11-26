import tensorflow as tf
import tensorflow.keras as keras
import os
import argparse

import numpy as np
import PIL.Image as pilimg
import matplotlib.pyplot as plt

# # Test
# img_path = os.path.join(os.getcwd(), "dataset", "abnormal", "10.jpg")
# img = pilimg.open(img_path)

# plt.imshow(img)
# plt.show()

# img = np.array(img)
# img = img[np.newaxis, ...]

# Argument set
parser = argparse.ArgumentParser(
    description="Simple 'argparse' demo application")
parser.add_argument('--img', dest='img_path', required=True,
                    help='Classfication image file path')

args = parser.parse_args()

# Exist check image file
if (not os.path.isfile(args.img_path)):
    print('Image file not found... [{}]'.format(args.img_path))

# Image load
img = keras.preprocessing.image.load_img(args.img_path, target_size=[512, 512])
img = keras.preprocessing.image.img_to_array(img)
img = img[tf.newaxis, ...]

# Model load
model_path = os.path.join(os.getcwd(), '1')

if not os.path.isdir(model_path):
    model_files = os.listdir(model_path)

    if len(model_files) == 0:
        print('Model not found....')
        exit(1)


model = keras.models.load_model(model_path)

p = model.predict(img)
result = 'abnormal' if p < 0.5 else 'normal'
print('predict:', p[0], result)
