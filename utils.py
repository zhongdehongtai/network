import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def convert_to_onehot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def Normalization(X):
    X=X-np.average(X)
    X=X/X.max()
    return X


def compute_cost(AL, Y):
    m = Y.shape[0]
    # Compute loss from AL and y
    logprobs = np.multiply(np.log(AL), Y) + np.multiply(1 - Y, np.log(1 - AL))
    # cross-entropy cost
    cost = - np.sum(logprobs) / m
    cost = np.squeeze(cost)
    return cost