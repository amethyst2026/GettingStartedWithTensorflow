import tensorflow as tf
import keras
from keras import datasets, regularizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from tensorflow.data import Dataset
import tensorflow.keras.datasets


mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[..., tf.newaxis].astype("float32")
train_labels = train_labels[..., tf.newaxis].astype("int32")
test_labels = test_labels[..., tf.newaxis].astype("int32")
test_images = test_images[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_labels)).shuffle(1000).batch(32)

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv2 = Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal', use_bias=True, strides=(2,2),kernel_regularizer=regularizers.l2(0.01),bias_initializer='he_normal',bias_regularizer=regularizers.l2(0.01))
        self.dropout2 = Dropout(0.5)
        self.batchnorm2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))
        self.flatten2 = Flatten()
        self.dense4 = Dense(512,activation='relu',kernel_initializer='he_normal')
        self.dense5 = Dense(512,activation='relu',kernel_initializer='he_normal')
        self.dense6 = Dense(10, activation='relu',kernel_initializer='he_normal')

    def call(self, x):

        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        x = self.pool2(x)
        x = self.flatten2(x)
        x = self.dense5(x)
        x = self.dense4(x)
        x = self.dense6(x)
        return x
class ConvLayer2(MyModel):
    def __init__(self):
        super(ConvLayer2, self).__init__()
        self.conv3 = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), bias_initializer='he_normal',use_bias = True, strides=(2,2))
        self.dropout3 = Dropout(0.5)
        self.batchnorm3 = BatchNormalization()
        self.pool3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))
        self.flatten3 = Flatten()
        self.dense7 = Dense(512,activation='relu',kernel_initializer='he_normal')
        self.dense8 = Dense(512,activation='relu',kernel_initializer='he_normal')
        self.dense9 = Dense(10, activation='relu',kernel_initializer='he_normal')

    def call(self, x):
        x = self.conv3(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)
        x = self.pool3(x)
        x = self.flatten3(x)
        x = self.dense7(x)
        x = self.dense8(x)
        x = self.dense9(x)

        return x
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum_over_batch_size')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

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