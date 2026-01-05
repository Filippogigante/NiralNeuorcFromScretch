import numpy as np

class ActivationFunctions():

    def forward(self, x):
        raise NotImplementedError 
        
    def backward(self, x):
        raise NotImplementedError


class relu(ActivationFunctions):

    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return ((x > 0).astype(float))
    
class tanh(ActivationFunctions):

    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - x ** 2.0
    
class identity(ActivationFunctions):

    def forward(self, x):
        return x
    
    def backward(self, x):
        return 1.0