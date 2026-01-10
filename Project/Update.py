import numpy as np

class update_params:
    def update(self, x):
        raise NotImplementedError 

class std_update(update_params):
    def __init__(self, eta):
        self.eta = eta

    def update(self, Layer):
        dW, d_bias = Layer.get_deltas()
        W = Layer.get_weights()
        b = Layer.get_bias()

        W = W - self.eta * dW
        b = b - self.eta * d_bias
        
        Layer.set_weights(W)
        Layer.set_bias(b)

        return W, b
    
class momentum_update(update_params):
    def __init__(self, eta, alpha):
        self.eta = eta
        self.alpha = alpha

    def update(self, Layer):
        dW, d_bias = Layer.get_deltas()
        W = Layer.get_weights()
        b = Layer.get_bias()
        vW = Layer.get_vW()
        vb = Layer.get_vb()

        vW = self.alpha * vW + (1 - self.alpha) * dW
        vb = self.alpha * vb + (1 - self.alpha) * d_bias
        W = W - self.eta * vW
        b = b - self.eta * vb

        Layer.set_vW(vW)
        Layer.set_vb(vb)
        Layer.set_weights(W)
        Layer.set_bias(b)

        return W, b