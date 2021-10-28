from __future__ import absolute_import, division, print_function, unicode_literals

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
# from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

import os, ssl
from skimage.transform import resize

# GPU error 조치
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# SSL Error 방지
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

print('Tensorflow version :', tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Global parameter
batch_size = 64
epochs = 100

img_width = 28 * 1
img_height = img_width

data_augmentation = False


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.reshape(img, [img_width, img_height])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 이미지 전처리
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.reshape(train_images, [train_images.shape[0], train_images.shape[1], train_images.shape[2], 1])
test_images = np.reshape(test_images, [test_images.shape[0], test_images.shape[1], test_images.shape[2], 1])

# Image resize
if img_width != 28:
    print('Image resizing...')
    train_images = resize(train_images, [60000, img_width, img_height, 1])
    test_images = resize(test_images, [10000, img_width, img_height, 1])

print('train shape :', train_images.shape, train_labels.shape)
print('test shape :', test_images.shape, test_labels.shape)

# Model 생성
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, [3, 3], input_shape=[img_width, img_height, 1], activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPool2D([2, 2], strides=2, padding='same'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPool2D([2, 2], strides=2, padding='same'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPool2D([2, 2], strides=2, padding='same'))
model.add(tf.keras.layers.Dropout(0.3))

if img_width >= 56:
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

if data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    # set input mean to 0 over the dataset
                    featurewise_center=False,
                    # set each sample mean to 0
                    samplewise_center=False,
                    # divide inputs by std of dataset
                    featurewise_std_normalization=False,
                    # divide each input by its std
                    samplewise_std_normalization=False,
                    # apply ZCA whitening
                    zca_whitening=False,
                    # epsilon for ZCA whitening
                    zca_epsilon=1e-06,
                    # randomly rotate images in the range (deg 0 to 180)
                    rotation_range=0,
                    # randomly shift images horizontally
                    width_shift_range=0.1,
                    # randomly shift images vertically
                    height_shift_range=0.1,
                    # set range for random shear
                    shear_range=0.,
                    # set range for random zoom
                    zoom_range=0.,
                    # set range for random channel shifts
                    channel_shift_range=0.,
                    # set mode for filling points outside the input boundaries
                    fill_mode='nearest',
                    # value used for fill_mode = "constant"
                    cval=0.,
                    # randomly flip images
                    horizontal_flip=True,
                    # randomly flip images
                    vertical_flip=False,
                    # set rescaling factor (applied before any other transformation)
                    rescale=True,
                    # set function that will be applied on each input
                    preprocessing_function=None,
                    # image data format, either "channels_first" or "channels_last"
                    data_format=None,
                    # fraction of images reserved for validation (strictly between 0 and 1)
                    validation_split=0.0)

    train_set = train_datagen.fit(train_images)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=batch_size),
                        validation_data=(test_images, test_labels),
                        epochs=epochs, verbose=2, workers=8, use_multiprocessing=True)

else:
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
              epochs=epochs, batch_size=batch_size, verbose=2)

print('Training Successed.')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('테스트 정확도:', test_acc)

predictions = model.predict(test_images)
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

