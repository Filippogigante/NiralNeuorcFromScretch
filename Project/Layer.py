import numpy as np
from Activations import *
from Initializer import *


class Layer:

    """def repr(self):
        return "classe"
        """

    def __init__(self, dim_input, dim_output, activation, initializer):

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.initializer = None
        self.activation = None
        self.dW = None
        self.d_bias = None
        self.net = None
        self.o = None

        activation_map = {
            "relu": relu,        
            "tanh": tanh,
            "identity": identity
        }

        initializer_map = {
            "standard": std_initializer
        }

        if isinstance(activation, str):
            try:
                self.activation = activation_map[activation.lower()]() 
            except KeyError:
                raise ValueError(f"Attivazione '{activation}' non supportata. Usa: {list(activation_map.keys())}")
        else:
            raise BaseException("activation must be a string")
        
        if isinstance(initializer, str):
            try:
                self.initializer = initializer_map[initializer.lower()]() 
            except KeyError:
                raise ValueError(f"Attivazione '{initializer}' non supportata. Usa: {list(initializer_map.keys())}")
        else:
            raise BaseException("initializer must be a string")

        self.W, self.b, self.vW, self.vb  = self.initializer.init_weights()

    def get_weights(self):
        return self.W
    
    def set_weights(self, weights):
        self.W = weights

    def get_bias(self):
        return self.b
    
    def set_bias(self, bias):
        self.b = bias

    def get_vW(self):
        return self.vW
    
    def set_vW(self, vW):
        self.vW = vW

    def get_vb(self):
        return self.vb
    
    def set_vb(self, vb):
        self.vb = vb

    def forward_pass(self, input):
        self.net = input.dot(self.W) + self.b
        self.o = self.activation(self.net)
        return self.o

    def backward_pass(self, delta, input, m):
        ## Restituisce il deltaW e il d_bias del layer corrente e il delta del layer precedente
        delta = delta * self.activation.backward(self.net)
        self.dW = (1 / m) * delta.dot(input.T)
        self.d_bias = (1 / m) * np.sum(delta , axis = 1 , keepdims = True)
        self.delta = ((self.W).T).dot(delta)
        return self.delta
    
    def get_deltas(self):
        return self.dW, self.d_bias