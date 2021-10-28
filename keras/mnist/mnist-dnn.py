# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

nb_epochs = 100
nb_batch = 64

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data() # we use only train set
X_train = (X_train / 255.0) * 2.0 - 1.0
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = dataset.take(len(dataset) * 70 // 100).shuffle(1024, reshuffle_each_iteration=True).batch(nb_batch)
valid_dataset = dataset.skip(len(dataset) * 70 // 100).batch(nb_batch)


def getModel(dropout_rate=0.0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


model = getModel(0.5)
model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
hist = model.fit(train_dataset, epochs=nb_epochs, validation_data=valid_dataset, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:{:.5f}'.format(score[0]))
print('Test accuracy: {:.3f}'.format(score[1]))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(hist.history['loss'], 'r-', label='loss')
plt.plot(hist.history['val_loss'], 'b-', label='val_loss')
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'], 'r-', label='accuracy')
plt.plot(hist.history['val_accuracy'], 'b-', label='val_accuracy')
plt.scatter(100, score[1], label='test_accuracy')
plt.legend()
plt.show()



