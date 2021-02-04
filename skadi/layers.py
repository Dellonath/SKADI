from skadi.activations import activation
from skadi.auxiliary import tools

class Layer():

    '''
    Layer class used to create a common dense layer with neurons.

    Attributes:
        num_units: return the number of neurons of this layer
        weights: return the layer weights
        biases: return the layer biases

    Hyperparameters:
        units: int, default 16
            Number of neurons of this layer.

        activation_function: str: default 'linear'
            Select the activation function of this layer neurons.
            
            Possible activation functions:
                'linear': linear                        f(x) = x
                'binary_step': binary step              f(x) = 0 if x < 0 else 1
                'relu': rectified linear unit (reLU)    f(x) = max(0, x)
                'leaky_relu': leaky-reLU                f(x) = alpha * x if x < 0 else x
                'elu': exponential linear unit (eLU)    f(x) = alpha * (exp(x) - 1) if x <= 0 else x
                'softmax': softmax                      f(x) = exp(x) / sum{i to n} (exp(x_i))        
                'sigmoid': sigmoid                      f(x) = 1.0 / (1.0 + exp(-x))
                'tanh': tanh                            f(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
                'soft_plus': soft plus                  f(x) = log(1 + exp(x))
        
        regularization: tuple, default ('l2', 0.0)
            Application of regularization[0] type in this Layer. 

                Possible regularization[0] value (regularization type):
                'l1': regularization L1, this regularization try to approach the weights to zero. This technique 
                select the more important features of the sample. Apply only if have overfitting problem.
                    
                    Cost = cost_function(y, w, b) + (regularization[1] * Σ ||w||) / num_samples

                'l2': regularization L2 or weight decay, this technique makes the network match all the attributes of 
                the samples in relation to the importance for learning. It is used more than the L1. Apply only if 
                have overfitting problem.
                    
                    Cost = cost_function(y, w, b) + (regularization[1] * Σ(w_i**2) ) / 2*num_samples

                Possible regularization[1] value (lambda):
                Float value. Often between 0 and 1.
    '''
 
    def __init__(self, units = 16, activation_function = 'linear', regularization = ('l2', 0.0)):
        
        self.num_units = units
        self.weights = None
        self.biases = None

        self._input_lenght = None
        self._input = None
        self._activation = activation[activation_function]
        self._regularization, self._reg_lambda = tools[regularization[0]], regularization[1]
        self._droped = 1
        self._summation, self._activated = None, None
        self._dweights, self._pre_dweights, self._dbiases = None, 0, None
        
class Dropout():

    '''
    Layer class used to create a Droput type layer with neurons.

    Attributes:
        num_units: return the number of neurons of this layer
        weights: return the layer weights
        biases: return the layer biases

    Hyperparameters:
        units: int, default 16
            Number of neurons of this layer.

        activation_function: str: default 'linear'
            Select the activation function of this layer neurons.
            
            Possible activation functions:
                'linear': linear                        f(x) = x
                'binary_step': binary step              f(x) = 0 if x < 0 else 1
                'relu': rectified linear unit (reLU)    f(x) = max(0, x)
                'leaky_relu': leaky-reLU                f(x) = alpha * x if x < 0 else x
                'elu': exponential linear unit (eLU)    f(x) = alpha * (exp(x) - 1) if x <= 0 else x
                'softmax': softmax                      f(x) = exp(x) / Σ(exp(x_i))  
                'sigmoid': sigmoid                      f(x) = 1.0 / (1.0 + exp(-x))
                'tanh': tanh                            f(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
                'soft_plus': soft plus                  f(x) = log(1 + exp(x))
        
        p: float, default 0.3
            Neurons rate that must be dropped in this layer. The float interval is between 0 and 1.

        regularization: tuple, default ('l2', 0.0)
            Application of regularization[0] type in this Layer. 

                Possible regularization[0] value (regularization type):
                'l1': regularization L1, this regularization try to approach the weights to zero. This technique 
                select the more important features of the sample. Apply only if have overfitting problem.
                    
                    Cost = cost_function(y, w, b) + (regularization[1] * Σ ||w||) / num_samples

                'l2': regularization L2 or weight decay, this technique makes the network match all the attributes of 
                the samples in relation to the importance for learning. It is used more than the L1. Apply only if 
                have overfitting problem.
                    
                    Cost = cost_function(y, w, b) + (regularization[1] * Σ(w_i**2) ) / 2*num_samples

                Possible regularization[1] value (lambda):
                Float value. Often between 0 and 1.
    '''
    
    def __init__(self, units = 16, activation_function = 'linear', p = 0.3, regularization = ('l2', 0.0)):
        
        self.num_units = units
        self.weights = None
        self.biases = None
        
        self._input_lenght = None
        self._input = None
        self._prob = p
        self._activation = activation[activation_function]
        self._regularization, self._reg_lambda = tools[regularization[0]], regularization[1]
        
        self._summation, self._activated = None, None
        self._dweights, self._pre_dweights, self._dbiases = None, 0, None
        self._droped = None
