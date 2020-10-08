import numpy as np
import tensorflow as tf ### For data only ###

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
num_classes = len(np.unique(train_labels))

train_labels_original = train_labels
test_labels_original = test_labels
### parse both sets to vectors ###
train_images = train_images.reshape(-1, train_images.shape[1] * train_images.shape[2])
test_images = test_images.reshape(-1, test_images.shape[1] * test_images.shape[2])

train_labels = tf.one_hot(train_labels, depth=num_classes)
test_labels = tf.one_hot(test_labels, depth=num_classes)

