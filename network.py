import numpy as np
from utils import compute_cost
import matplotlib.pyplot as plt
from activation_func import relu, relu_backward, sigmoid, sigmoid_relu, sigmoid_backward, sigmoid_relu_backward,prelu,prelu_backward

learnable_acti_func = {"prelu"}


class network(object):
    def __init__(self, nn_acchitecture, flag=True, alpha=0.5, belta=0.5, gamma=0.5):
        self.nn_architecture = nn_acchitecture
        self.alpha = alpha
        self.belta = belta
        self.gamma = gamma
        self.flag = flag
        self.result = []

    def initialize_parameters(self, seed=3):
        np.random.seed(seed)
        parameters = {}
        number_of_layers = len(self.nn_architecture)
        for l in range(1, number_of_layers):
            parameters['W' + str(l)] = np.random.randn(
            self.nn_architecture[l]["layer_size"],
            self.nn_architecture[l-1]["layer_size"]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.nn_architecture[l]["layer_size"], 1))
            activation = self.nn_architecture[l]["activation"]
            if activation in learnable_acti_func:
                parameters['alpha'+str(l)] = np.zeros((self.nn_architecture[l]["layer_size"], 1))*self.gamma
        return parameters

    def L_model_forward(self, X, parameters):
        forward_cache = {}
        A = X
        forward_cache['A' + str(0)] = A
        number_of_layers = len(self.nn_architecture)
        for l in range(1, number_of_layers):
            A_prev = A
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            activation =self.nn_architecture[l]["activation"]
            alpha = None
            if activation in learnable_acti_func:
                alpha = parameters['alpha'+str(l)]
            Z, A = self.linear_activation_forward(A_prev, W, b, activation, alpha)
            forward_cache['Z' + str(l)] = Z
            forward_cache['A' + str(l)] = A
            AL = A
        return AL, forward_cache

    def linear_activation_forward(self, A_prev, W, b, activation, alpha=None):
        if activation =="sigmoid":
            Z = self.linear_forward(A_prev, W, b)
            A = sigmoid(Z)
        elif activation =="relu":
            Z = self.linear_forward(A_prev, W, b)
            A = relu(Z)
        elif activation == "prelu":
            Z = self.linear_forward(A_prev, W, b)
            A = prelu(Z, alpha)
        return Z, A

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        return Z

    def L_model_backward(self, AL, Y, parameters, forward_cache):
        grads = {}
        number_of_layers = len(self.nn_architecture)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # afterthis line, Y is the same shape as AL
        # Initializing thebackpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA_prev = dAL
        for l in reversed(range(1, number_of_layers)):
            dA_curr = dA_prev
            activation =self.nn_architecture[l]["activation"]
            alpha_curr = None
            if activation in learnable_acti_func:
                alpha_curr = parameters['alpha'+str(l)]
            W_curr = parameters['W' +str(l)]
            Z_curr = forward_cache['Z' +str(l)]
            A_prev = forward_cache['A' +str(l-1)] ##str(l-1)
            dA_prev, dW_curr, db_curr, dalpha = self.linear_activation_backward(dA_curr, Z_curr, A_prev, W_curr, activation, alpha_curr)
            grads["dW" +str(l)] = dW_curr
            grads["db" +str(l)] = db_curr
            if id(dalpha)==id(None):
                pass
            else:
                grads['dalpha'+str(l)] = dalpha
        return grads

    def linear_activation_backward(self, dA, Z, A_prev, W, activation, alpha_curr):
        if activation =="relu":
            dZ = relu_backward(dA, Z)
            dA_prev, dW, db, dalpha = self.linear_backward(dZ, A_prev, W)
        elif activation =="sigmoid":
            dZ = sigmoid_backward(dA, Z)
            dA_prev, dW, db, dalpha = self.linear_backward(dZ, A_prev, W)
        elif activation == "prelu":
            dZ, da = prelu_backward(dA, Z, alpha_curr)
            dA_prev, dW, db, dalpha = self.linear_backward(dZ, A_prev, W, da)
        return dA_prev, dW, db, dalpha

    def linear_backward(self, dZ, A_prev, W, dalpha=None):
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db, dalpha

    def update_parameters(self, parameters, grads, learning_rate, grads_old):
        L = len(self.nn_architecture)
        for l in range(1, L):
            if not self.flag:
                parameters["W" +str(l)] -= learning_rate *grads["dW" + str(l)]
                parameters["b" +str(l)] -= learning_rate *grads["db" + str(l)]
                if activation in learnable_acti_func:
                    parameters['alpha' + str(l)] -= learning_rate * grads['dalpha' + str(l)]
            else:
                if not grads_old:
                    parameters["W" +str(l)] -= learning_rate *grads["dW" + str(l)]
                    parameters["b" +str(l)] -= learning_rate *grads["db" + str(l)]
                    activation = self.nn_architecture[l]["activation"]
                    if activation in learnable_acti_func:
                        parameters['alpha'+str(l)] -= learning_rate*grads['dalpha'+str(l)]
                else:
                    parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)] + self.alpha*grads_old['dW' + str(l)]
                    parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)] + self.alpha*grads_old['db' + str(l)]
                    activation = self.nn_architecture[l]["activation"]
                    if activation in learnable_acti_func:
                        parameters['alpha'+str(l)] -= learning_rate*grads['dalpha'+str(l)] + self.belta*grads_old['dalpha'+str(l)]
        return parameters

    def L_layer_model(self, X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        np.random.seed(1)
        # keep track of cost
        costs = []
        steps = []
        # Parameters initialization.
        parameters = self.initialize_parameters()
        # Loop (gradient descent)
        grads_old = None
        for i in range(0, num_iterations):
            # Forward propagation:[LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, forward_cache = self.L_model_forward(X, parameters)
            # Compute cost.
            cost = compute_cost(AL, Y)
            # Backward propagation.
            grads = self.L_model_backward(AL, Y, parameters, forward_cache)
            # Update parameters.
            parameters = self.update_parameters(parameters, grads, learning_rate, grads_old)
            grads_old = grads
            # Print the cost every 100training example
            if print_cost and i % 4 ==0:
                print("Cost afteriteration %i: %f" % (i, cost))
                steps.append(i)
                costs.append(cost)
        self.result.append((steps, costs))
        return parameters

    def predict(self, X, y, parameters):
        m = X.shape[1]
        probas, caches = self.L_model_forward(X, parameters)
        predict_label = np.argmax(probas, axis=0)
        print("预测准确度: " + str(np.sum((predict_label == y) / m)))
        return np.sum((predict_label == y) / m)