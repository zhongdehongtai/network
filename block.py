import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import keras
from activation_func import relu, relu_backward, sigmoid, sigmoid_backward, sigmoid_relu, sigmoid_relu_backward
from utils import convert_to_onehot, Normalization
from network import network
import pickle


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# Figure out the dimensions and shapes of the problem
m_train = len(train_images)
m_test = len(test_images)
num_px = train_images.shape[1]

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ")")
print("train_images shape: " + str(train_images.shape))
print("train_labels shape: " + str(train_labels.shape))
print("test_images shape: " + str(test_images.shape))
print("test_labels shape: " + str(test_labels.shape))

train_images_flatten = train_images.reshape(train_images.shape[0], -1).T
test_images_flatten = test_images.reshape(test_images.shape[0], -1).T

train_labels_onehot = convert_to_onehot(train_labels, 10)
test_labels_onehot = convert_to_onehot(test_labels, 10)

print("train_images_flatten shape: " + str(train_images_flatten.shape))
print("train_labels_onehot shape: " + str(train_labels_onehot.shape))
print("test_images_flatten shape: " + str(test_images_flatten.shape))
print("test_labels_onehot shape: " + str(test_labels_onehot.shape))

train_set_x = train_images_flatten / 255.0
test_set_x = test_images_flatten / 255.0

nn_architecture = [
{"layer_size": 784, "activation": "none"}, # input layer
{"layer_size": 128, "activation": "sigmoid"},
{"layer_size": 10, "activation": "sigmoid"},
]

if __name__ == "__main__":
    network_now = network(nn_architecture, True)
    train_x = train_set_x[:, 0:10000]
    train_y = train_labels_onehot[:, 0:10000]
    test_x = test_set_x
    test_y = test_labels
    parameters = network_now.L_layer_model(train_x, train_y, learning_rate=0.1, num_iterations=1500, print_cost=True)
    pickle.dump(network_now.result, fw, -1)
    acc = network_now.predict(test_x, test_y, parameters)
    print('---------')
    print(acc)
    print('---------')
