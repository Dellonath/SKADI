import numpy as np

def linear(x, diff = False):
    if diff:
        return np.ones_like(x)
    return x       

def binary_step(x, diff = False):
    if diff:
        return np.where(x != 0, 0, np.inf)
    return np.where(x < 0, 0, 1)

def relu(x, diff = False):
    if diff:
        x = np.where(x <= 0, 0, 1)
        return np.maximum(0, x)
    return np.maximum(0, x)     

def softmax(x, y_oh = None, diff = False):
    if diff:
        y_pred = softmax(x)
        k = np.nonzero(y_pred * y_oh)
        pk = y_pred[k]
        y_pred[k] = pk * (1 - pk)
        return y_pred
    exp = np.exp(x)
    return exp / np.sum(exp, axis = 1, keepdims = True)

def sigmoid(x, diff = False):
    if diff:
        return sigmoid(x) * (1.0 - sigmoid(x)) 
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x, diff = False):
    if diff:
        return 1 - tanh(x)**2
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def leaky_relu(x, diff = False):
    alpha = 0.001
    if diff:
        return np.where(x <= 0, alpha, 1)
    return np.where(x <= 0, alpha * x, x)

def elu(x, diff = False):
    alpha = 1.0
    if diff:
        return np.where(x <= 0, elu(x) + alpha, 1)
    return np.where(x <= 0, alpha * (np.exp(x) - 1), x)

def soft_plus(x, diff = False):
    if diff:
        return sigmoid(x)
    return np.log(1 + np.exp(x))

activation = {
    'linear': linear,
    'binary_step': binary_step,
    'relu': relu,
    'softmax': softmax,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'leaky_relu': leaky_relu,
    'elu': elu,
    'soft_plus': soft_plus
}
