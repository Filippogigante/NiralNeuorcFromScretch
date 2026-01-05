import numpy as np

class loss:

    def forward_loss(self, x, y):
        raise NotImplementedError
    
    def backward_loss(self, x, y):
        raise NotImplementedError
        
class mse(loss):

    def forward_loss(self, output, target):
        return np.mean(np.sum(np.power(output - target, 2), axis=0, keepdims=True))
    
    def backward_loss(self, output, target):
        return output - target
    
class mee(loss):

    def forward_loss(self, output, target):
        return np.mean(np.sqrt(np.sum(np.power(output - target, 2), axis=0, keepdims=True)))
    
    def backward_loss(self, output, target):
        return (output - target) / self.forward_loss(output, target)