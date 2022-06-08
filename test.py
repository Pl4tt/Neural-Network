import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mnist import MNIST
import matplotlib.pyplot as plt


mndata = MNIST("digit_data")

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

train_images = mndata.process_images_to_numpy(train_images)
train_labels = mndata.process_images_to_numpy(train_labels)
test_images = mndata.process_images_to_numpy(test_images)
test_labels = mndata.process_images_to_numpy(test_labels)
# print(np.shape(train_images), np.shape(train_labels), np.shape(test_images), np.shape(test_labels))  # (60000, 784) (60000,) (10000, 784) (10000,)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="sigmoid"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()],)
model.fit(train_images, train_labels, epochs=40)

model.save("tensorflow_model.h5")

# length = len(test_images)
# predictions = np.array([np.where(arr == max(arr)) for arr in model.predict(test_images[:length])]).reshape((length,))
# print(predictions, test_labels[:length])

results = model.evaluate(test_images, test_labels)
print(results)