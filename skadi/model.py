import numpy as np
import pickle as pkl
from skadi.auxiliary import tools
from skadi.auxiliary import split_data
from skadi.auxiliary import prediction
from skadi.auxiliary import acc
from skadi.auxiliary import loss_reg
from skadi.auxiliary import prepare_layer
from skadi.auxiliary import verbose_progess
from skadi.auxiliary import loss_stop_checker
from skadi.auxiliary import learning_rate_decay
from skadi.costs import cost
from skadi.layers import Layer
from skadi.layers import Dropout

class Model():

    '''
    Model class to initialize the neural network. Through this class it will be possible 
    to add layers of neurons and determine some hyperparameters that will be used in training.

    Attributes:
        history: dict 
            return the history of valitation accuracy ('val_acc'), train loss ('loss) and validation loss ('val_loss').

    Hyperparameters: 

        cost_function: str, default 'mse'
            Cost function to be used. Utilized to calculate the loss between prediction and targets.
            
            Possible cost functions:
                'mse': mean squared error
                'mae': mean absolute error
                'binary_cross_entropy': binary cross entropy
                'sigmoid_cross_entropy': sigmoid cross entropy
                'softmax_neg_log_likelihood': softmax function + negative log-likelihood

        learning_rate: float, default 0.001
            Learning rate is used to determin how much the neural network weights will be updated. Often values between 0 and 1.

        momentum: float, default 0.0
            Momentum is a technique that adds the previous gradient multiplied by a factor to the current gradient, 
            accelerating the descending gradient, making the network converge more quickly. This hiperparameter named 
            momentum is the factor.
                
                W(t) = W(t-1) - (learning_rate * ∂Cost/∂W + momentum * (W(t-1) - W(t-2)))

        batch: tuple, default ('sequential', None)
            This parameter split your data in N batches. The first value is the batch type and the second values is 
            how many samples in each batch.
            
            Possible batch[0] value (batch type):
                'sequential': the data will be separeted in sequential order (1, 2, 3), (4, 5, 6)...
                'shuffled': the data will be shuffled before being separated (5, 1, 3), (2, 4, 6)...
            
            Possible batch[1] value (number of batches):
                1: stochastic gradient descent, the neural network will train faster, but with great less quality
                1 < n < N: mini-batch gradient descent, the neural network will train a bit more faster, but with less quality
                None: batch gradient descent, the neural network takes longer to train, but more efficiently
            
        lr_decay: tuple, default (None, None, None)
            Learning Rate decay is a technique that change the learning rate according to the epoch.

            Possible lr_decay[0] value (learning rate decay type):
                'time-based': the learning rate will be started with 1.0 and will decrease according to the epoch
                    learning_rate = 1.0 / (1.0 + decay_rate * epoch)
                'exponential': the learning rate decreases exponentially according to the epoch
                    learning_rate = learning_rate * (decay_rate ** epoch)
                'staircase': the learning rate will be decreased based on staircase.
                    learning_rate = learning_rate * decay_rate ** (epoch // decay_step)

            Possible lr_decay[1] value (decay rate):
                Must be a float value. Often used between 0 and 1.

            Possible lr_decay[2] value (decay step):
                Must be a integer and between 1 and N (number of epochs).

    Methods: 

        add(self, new_layer):
            Adds a new layer in the neural network.

            Parameters: 
                new_layer (Layer instance): must be a instance of skadi.layers.Layer() or skadi.layers.Dropout()
        
        predict(self, x):
            Prediction when gived x data.

            Parameters: 
                x: data values, must be same type and shape that train data
        
        accuracy(self, y, y_pred):
            Return the Model score.
            
            Parameters: 
                y: targets
                y_pred: predicted data

        fit(self, x, y, epochs = 100, validation_rate = 0.2, verbose = None, loss_stop = (None, None)):
            Train the Model using x and y data.

            Parameters:
                x: data
                y: data targets
                epochs: number of epochs
                validation_rate: splits the data in train and validation, is not possible train without samples for model validation 
                verbose: verbose progress, for each verbose time will have a train summary
                loss_stop: stop the train if the loss is less than loss_stop[0] by loss_stop[1] times

        save(self, patch):
            Save the Model in a pickle archive.

            Parameters:
                patch: str that describe the directory

        load(patch):
            Load the Model in a pickle archive.

            Parameters:
                patch: str that describe the directory
    '''

    def __init__(self, cost_function = 'mse', learning_rate = 0.001, momentum = 0.0, batch = ('sequential', None), lr_decay = (None, None, None)):
        
        self.history = {'val_acc': [], 'loss': [], 'val_loss': []}
        self.__cost = cost[cost_function]

        self.__lr_decay = lr_decay
        self.__batch, self.__batch_size = tools[batch[0]], batch[1]
        self.__learning_rate = self.__lr_init = learning_rate
        self.__layers = []
        self.__momentum = momentum

    def add(self, new_layer):
        self.__layers.append(new_layer)
        prepare_layer(new_layer, self.__layers)
        
    def predict(self, x):
        return prediction(x, self.__cost, self.__feedforward)
    
    def accuracy(self, y, y_pred):
        return acc(y, y_pred, self.__cost)
    
    def fit(self, x, y, epochs = 100, validation_rate = 0.2, verbose = None, loss_stop = (None, None)):
        
        x_train, x_test, y_train, y_test = split_data(x, y, validation_rate)
        loss_score = 0

        for epoch in range(epochs + 1):
            self.__learning_rate = learning_rate_decay(self.__lr_init, epoch, self.__lr_decay[0], self.__lr_decay[1], self.__lr_decay[2])
            for x_train, y_train in self.__batch(x_train, y_train, self.__batch_size):
                y_pred_val = self.__feedforward(x_test)
                y_pred = self.__feedforward(x_train, True)
                self.__backpropagation(y_train, y_pred)

            self.history['loss'].append(self.__cost(y_train, y_pred) + loss_reg(y_train, self.__layers))
            self.history['val_loss'].append(self.__cost(y_test, y_pred_val))
            self.history['val_acc'].append(self.accuracy(y_test, self.predict(x_test)))
            
            verbose_progess(verbose, epoch, epochs, self.history)
            
            loss_score = loss_stop_checker(epoch, epochs, loss_stop, loss_score + 1, self.history)
            if loss_score == -1: break

    def save(self, path):
        pkl.dump(self, open(path, 'wb'), -1)
    
    def load(path):
        return pkl.load(open(path, 'rb'))

    def __feedforward(self, x, training = False):

        self.__layers[0]._input = x

        for current_layer, next_layer in zip(self.__layers, self.__layers[1:] + [Layer(0)]):
            current_layer._summation = np.dot(current_layer._input, current_layer.weights.T) + current_layer.biases
            current_layer._activated = current_layer._activation(current_layer._summation)

            if isinstance(current_layer, Dropout) and training:
                prob = current_layer._prob
                current_layer._droped = np.random.binomial(1, 1 - prob, current_layer._summation.shape) / (1 - prob)
                next_layer._input = current_layer._activated * current_layer._droped
            else:
                next_layer._input = current_layer._activated
        
        return self.__layers[-1]._activated
    
    def __backpropagation(self, y, y_pred):
        
        last_delta = self.__cost(y, y_pred, diff=True)
        
        for layer in reversed(self.__layers):  
            dactivation = layer._activation(layer._activated, diff=True) * last_delta * layer._droped
            last_delta = np.dot(dactivation, layer.weights)
            layer._dweights = np.dot(dactivation.T, layer._input)
            layer._dbiases = 1.0 * dactivation.sum(axis=0, keepdims=True)
        
        for layer in reversed(self.__layers):
            layer._dweights = layer._dweights + layer._regularization(layer.weights, layer._reg_lambda, diff = True) / y.shape[0]
            layer._pre_dweights = - self.__learning_rate * layer._dweights + self.__momentum * layer._pre_dweights
            layer.weights = layer.weights + layer._pre_dweights
            layer.biases = layer.biases - self.__learning_rate * layer._dbiases
            
