# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

np.random.seed(1337)  # for reproducibility

batch_size = 256
nb_classes = 10
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 28, 28


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
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

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Conv2D(32, [3, 3], input_shape=input_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(64, [3, 3], activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128, [3, 3], activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(625, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
