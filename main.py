import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


# Load the MNIST dataset
(train_images, train_labels),(test_images, test_labels)=datasets.mnist.load_data()

# Preprocessing: Normalize the pixel values to be between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

#reshape the images to (28, 28, 1) as they are greyscale
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#convert the labels to one-hot encoded format