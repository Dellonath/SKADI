import numpy as np
from skadi.activations import softmax
from skadi.activations import sigmoid
from skadi.costs import softmax_neg_log_likelihood
from skadi.costs import sigmoid_cross_entropy
from skadi.costs import binary_cross_entropy

def split_data(x, y, validation_rate):
    validation_rate = int(validation_rate * len(x))
    choices = np.random.choice(len(x), validation_rate, replace = False)
    x_test = x[choices]
    x_train = np.delete(x, choices, axis = 0)
    y_test = y[choices]
    y_train = np.delete(y, choices, axis = 0)
    
    return x_train, x_test, y_train, y_test  
    
def score(y, y_pred, classification = False):
    if classification:
        return np.count_nonzero((y_pred.round() == y).all(axis = 1)) / len(y)
    return np.sum(np.abs(y - y_pred))

def regularizationL1(weights, reg_lambda = 1e-2, diff = False):
    if diff:
        weights = [np.where(weight < 0, -1, weight) for weight in weights]
        return np.array([np.where(weight > 0, 1, weight) for weight in weights])
    return reg_lambda * np.sum([np.sum(np.abs(weight)) for weight in weights])

def regularizationL2(weights, reg_lambda = 1e-2, diff = False):
    if diff:
        return reg_lambda * weights
    return reg_lambda * 0.5 * np.sum(weights ** 2)

def batch_seq(x, y, batch_size = None):
    batch_size = len(x) if batch_size is None else batch_size
    num_batches = len(x) // batch_size
   
    for batch in range(num_batches):
        lenght = batch_size * batch
        yield (x[lenght:lenght + batch_size], y[lenght:lenght + batch_size])
        
def batch_shuffled(x, y, batch_size = None):
    index = np.random.permutation(range(len(x)))
    return batch_seq(x[index], y[index], batch_size)

def prediction(x, cost, fw):
    if cost == softmax_neg_log_likelihood:
        return softmax(fw(x))
    elif cost == sigmoid_cross_entropy:
        return sigmoid(fw(x)).round()
    return fw(x)

def acc(y, y_pred, cost):
    if cost in (sigmoid_cross_entropy, softmax_neg_log_likelihood, binary_cross_entropy):
        return score(y, y_pred, True)
    return score(y, y_pred)

def loss_reg(y, layers):
    loss_reg_weights = np.sum([layer._regularization(layer.weights, layer._reg_lambda) for layer in layers])
    return loss_reg_weights / y.shape[0]

def prepare_layer(new_layer, layers):
    new_layer._input_lenght = layers[layers.index(new_layer) - 1].num_units
    sigma = np.sqrt(2/(new_layer.num_units + new_layer._input_lenght))
    new_layer.weights = np.random.randn(new_layer.num_units, new_layer._input_lenght) * sigma
    new_layer.biases = np.random.randn(1, new_layer.num_units) * sigma

def verbose_progess(verbose, epoch, epochs, history):
    if verbose != None and epoch % verbose == 0:
        msg = f"epoch: {epoch:=5}/{epochs}  loss: {history['loss'][-1]:.10f}  val_acc: {history['val_acc'][-1]:.3f}"
        print(msg)

def loss_stop_checker(epoch, epochs, loss_stop, loss_score, history):
    if epoch > 0 and loss_stop[0] != None:
        if abs(history['loss'][-1] - history['loss'][-2]) < loss_stop[1]:
            if loss_score == loss_stop[0]:
                msg = f"loss_stop activated: {epoch:=5}/{epochs}  loss: {history['loss'][-1]:.10f}  val_acc: {history['val_acc'][-1]:.3f}"
                print(msg)
                return -1
        else:
            return 0
    return loss_score

def learning_rate_decay(lr, epoch, lr_decay_type, decay_rate, decay_step):
    if lr_decay_type == 'time-based':
        return 1.0 / (1.0 + decay_rate * epoch)
    elif lr_decay_type == 'exponential':
        return lr * (decay_rate**epoch)
    elif lr_decay_type == 'staircase':
        return lr * decay_rate ** (epoch // decay_step)
    return lr


tools = {
    'l1': regularizationL1,
    'l2': regularizationL2,
    'sequential': batch_seq,
    'shuffled': batch_shuffled,
}