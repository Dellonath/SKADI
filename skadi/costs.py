import numpy as np
from skadi.activations import softmax
from skadi.activations import sigmoid

def mse(y, y_pred, diff = False):
    if diff:
        return -(y - y_pred) / y.shape[0]
    return 0.5 * np.mean((y - y_pred)**2)

def mae(y, y_pred, diff = False):
    if diff:
        return np.where(y_pred > y, 1, -1) / y.shape[0]
    return np.mean(np.absolute(y - y_pred))

def binary_cross_entropy(y, y_pred, diff=False):
    if diff:
        return -(y - y_pred) / (y_pred * (1-y_pred) * y.shape[0])
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def sigmoid_cross_entropy(y, y_pred, diff=False):   
    y_sigmoid = sigmoid(y_pred)
    if diff:
        return -(y - y_sigmoid) / y.shape[0]
    return -np.mean(y * np.log(y_sigmoid) + (1 - y) * np.log(1 - y_sigmoid))

def neg_log_likelihood(y_oh, y_pred, diff = False):
    k = np.nonzero(y_pred * y_oh)
    pk = y_pred[k]
    if diff:
        y_pred[k] = -1.0 / pk
        return y_pred
    return np.mean(-np.log(pk))

def softmax_neg_log_likelihood(y_oh, y_pred, diff=False):
    y_softmax = softmax(y_pred)
    if diff:
        k = np.nonzero(y_pred * y_oh)
        dlog = neg_log_likelihood(y_oh, y_softmax, diff = True)
        dsoft = softmax(y_pred, y_oh, diff = True)
        y_softmax[k] = dlog[k] * dsoft[k]
        return y_softmax / y_softmax.shape[0]
    return neg_log_likelihood(y_oh, y_softmax)

cost = {
    'mse': mse, 
    'mae': mae,
    'binary_cross_entropy': binary_cross_entropy, 
    'sigmoid_cross_entropy' : sigmoid_cross_entropy,
    'softmax_neg_log_likelihood': softmax_neg_log_likelihood
}