import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def sigmoid(Z):
    S = 1 / (1 + np.exp(-Z))
    return S


def sigmoid_backward(dA, Z):
    S = sigmoid(Z)
    dS = S * (1 - S)
    return dA * dS


def relu(Z):
    R = np.maximum(0, Z)
    return R


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<0] = 0
    return dZ


def prelu(Z, alpha):
    R = np.array(Z.T, copy=True)
    indexs = np.where(R<0)
    for index, x in enumerate(indexs[0]):
        y = indexs[1][index]
        R[x, y] = alpha[y][0]*R[x][y]
    R = R.T
    return R




def tanh_relu(x, alpha=0.5):
    return np.where(np.greater(x, 0), x, np.tanh(alpha * x))


def tanh_relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)


def exp_relu(x, alpha=0.5):
    return np.where(np.greater(x, 0), x, np.exp(alpha * x) - 1)


def exp_relu_backward(dA, Z):
    pass


def sigmoid_relu(x, alpha=0.5):
    return np.where(np.greater(x, 0), x, 1 / (np.exp(-alpha * x) + 1) - 0.5)


def sigmoid_relu_backward(dA, Z, alpha):
    rate = 0.9
    dZ = np.array(dA, copy=True)
    dalpha = np.array(alpha,copy=True)
    indexs_1 = np.where(Z.T<0)
    indexs_2 = np.where(Z.T>=0)
    for index,x in enumerate(indexs_1[0]):
        y = indexs_1[1][index]
        dalpha_y_cur = dalpha[y]
        S = sigmoid(Z[x, y])
        dZ[x, y] = S * (1 - S)
        dalpha[y] = (1 - rate) * Z.T[x, y] * dZ[x, y] + rate * dalpha_y_cur
    for index, x in enumerate(indexs_2[0]):
        y = indexs_2[1][index]
        dZ[x, y] = 1
    return dZ


def prelu_backward(dA, Z, alpha):
    rate = 0.9
    dZ = np.array(dA.T, copy=True)
    dalpha = np.array(alpha, copy=True)
    indexs_1 = np.where(Z.T < 0)
    indexs_2 = np.where(Z.t>=0)
    for index, x in enumerate(indexs_1[0]):
        y = indexs_1[1][index]
        dalpha_y_cur = dalpha[y]
        dZ[x, y] = dalpha[y]
        dalpha[y] = (1-rate)*Z.T[x, y]*dZ[x, y]+rate*dalpha_y_cur
    for index, x in enumerate(indexs_2[0]):
        y = indexs_2[1][index]
        dZ[x, y] = 1
    return dZ.T, dalpha






