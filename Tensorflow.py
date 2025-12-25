from idlelib import history

import tensorflow as tf
import keras
from keras import Model, Sequential
from keras.src.backend.jax.nn import batch_normalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import pandas as pd
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import MaxPooling2D
import keras.src.layers.normalization.batch_normalization
from tensorflow.python.keras.saving.saved_model_experimental import sequential

from Tensorflow2 import mnist

mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[..., tf.newaxis].astype("float32")
train_labels = train_labels[..., tf.newaxis].astype("int32")
test_labels = test_labels[..., tf.newaxis].astype("int32")
test_images = test_images[..., tf.newaxis].astype("float32")


train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_labels)).shuffle(1000).batch(32)

class MyModel(Model):
  def __init__(self):
    super().__init__()
    self.conv1 = Conv2D(32, 3, activation='relu', padding='same')
    self.pool1 = MaxPooling2D(pool_size=(2, 2))
    self.dropout1 = Dropout(0.25)
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


  def __init__(self2):
      super().__init__()
      self2.conv2 = Conv2D(32, 3, activation='relu', padding='same')
      self2.pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))
      self2.flatten2 = Flatten()
      self2.d3 = Dense(128, activation='relu')
      self2.d4 = Dense(10)

  def call2(self2,x2):
    x = self2.conv2(x2)
    x = self2.dropout2(x2)
    x = self2.pool2(x2)
    x = self2.flatten2(x2)
    x = self2.d3(x2)
    return self2.d4(x2)

# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    for images, labels in train_ds:
        train_step(images,labels)

    for images, labels in test_ds:
        test_step(images,labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():0.2f}, '
        f'Accuracy: {train_accuracy.result() * 100:0.2f}, '
        f'Test Loss: {test_loss.result():0.2f}, '
        f'Test Accuracy: {test_accuracy.result() * 100:0.2f}'
    )