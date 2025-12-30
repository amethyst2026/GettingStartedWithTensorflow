import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D,BatchNormalization
from keras.optimizers import Adam
import keras as kr
from keras.models import Sequential
from keras.datasets import mnist
from keras.models import Model
import tensorflow as tf
import keras
from keras import datasets, regularizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from tensorflow.data import Dataset
import tensorflow.keras.datasets

(train_images, train_labels), (test_images, test_labels)= kr.datasets.boston_housing.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[..., tf.newaxis].astype("float32")
train_labels = train_labels[..., tf.newaxis].astype("int32")
test_labels = test_images[..., tf.newaxis].astype("int32")
test_images = test_labels[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(test_images).shuffle(10000).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(64,(3,3),padding = 'same',kernel_regularizer='l2(0.01)',bias_regularizer='l2(0.01)',bias_constraint='l2(0.01))',use_bias=True,kernel_initializer='he_normal',strides=(2,2), activation='relu',activity_regularizer='l2(0.01)')
        self.norm_layer = kr.layers.Normalization(axis=-1)
        self.batch_norm_layer = kr.layers.BatchNormalization(axis=-1)
        self.max_pooling_layer = kr.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.flatten_layer = kr.layers.Flatten()
        self.dropout_layer = kr.layers.Dropout(0.5)
        self.dense_layer = kr.layers.Dense(128, activation='softmax')
        self.dense_layer_2 = kr.layers.Dense(256, activation='softmax')
        self.dense_layer_3 = kr.layers.Dense(128, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = self.batch_norm_layer(x)
        x = self.max_pooling_layer(x)
        x = self.flatten_layer(x)
        x = self.dropout_layer(x)
        x = self.dense_layer(x)
        x = self.dense_layer_2(x)
        x = self.dense_layer_3(x)
        return x
model = MyModel

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum_over_batch_size')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images,training = True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss.update_state(t_loss)
  test_accuracy.update_state(labels, predictions)

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