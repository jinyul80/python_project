# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


np.random.seed(1337)  # for reproducibility

batch_size = 256
nb_classes = 10
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 28, 28


# the data, shuffled and split between train and test sets
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


def getModel(dropout_rate=0.0):
    model = Sequential()

    model.add(Conv2D(32, [3, 3], input_shape=input_shape, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, [3, 3], activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(128, [3, 3], activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(625, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


model = getModel(0.3)
model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:{:.5f}'.format(score[0]))
print('Test accuracy: {:.3f}'.format(score[1]))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], 'r-', label='loss')
plt.plot(hist.history['val_loss'], 'b-', label='val_loss')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], 'r-', label='accuracy')
plt.plot(hist.history['val_accuracy'], 'b-', label='val_accuracy')
plt.scatter(nb_epoch, score[1], label='test_accuracy')
plt.legend()
plt.show()