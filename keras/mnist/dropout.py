# -*- coding: utf-8 -*-
"""Dropout.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ebIe6CyEgly3d1cjq0nL_7_rV-lG8IrE
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mnist = tf.keras.datasets.mnist

(X, y), _ = mnist.load_data() # we use only train set
X = (X / 255.0) * 2.0 - 1.0
dataset = tf.data.Dataset.from_tensor_slices((X, y))
train_dataset = dataset.take(len(dataset) * 35 // 100).shuffle(1024, reshuffle_each_iteration=True).batch(32)
valid_dataset = dataset.skip(len(dataset) * 35 // 100).take(len(dataset) * 15 // 100).batch(32)
test_dataset = dataset.skip(len(dataset) * 50 // 100)

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

ITER_NUM = 5
EPOCHS = 50

hists = []
for iter in range(ITER_NUM):
    model = getModel()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    hist = model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset,verbose=0)
    hists.append(hist)

    print('Iter: {}, loss: {:.3f}, val_loss: {:.3f}'.format(iter+1, hist.history['loss'][-1], hist.history['val_loss'][-1]))

hists_dropout = []
for iter in range(ITER_NUM):
    model = getModel(0.5)
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    hist = model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset,verbose=0)
    hists_dropout.append(hist)

    print('Iter: {}, loss: {:.3f}, val_loss: {:.3f}'.format(iter+1, hist.history['loss'][-1], hist.history['val_loss'][-1]))

iters = len(hists)
epochs = len(hist.epoch)
train_loss = np.zeros((epochs,))
valid_loss = np.zeros((epochs,))
for h in hists:
    train_loss += np.array(h.history['loss']) / iters
    valid_loss += np.array(h.history['val_loss']) / iters

train_dropout_loss = np.zeros((epochs,))
valid_dropout_loss = np.zeros((epochs,))
for h in hists_dropout:
    train_dropout_loss += np.array(h.history['loss']) / iters
    valid_dropout_loss += np.array(h.history['val_loss']) / iters

plt.plot(train_loss, 'r-', label='dropout=0.0')
plt.plot(valid_loss, 'r--')
plt.plot(train_dropout_loss, 'b-', label='dropout=0.5')
plt.plot(valid_dropout_loss, 'b--')
plt.legend()
plt.show()

