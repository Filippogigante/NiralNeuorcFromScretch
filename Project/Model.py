import numpy as np
from Loss import *
from Update import *
from Layer import *

class Model:
    def __init__(self, eta, alpha, layers, update, loss, metric):
        self.layers = layers
        self.eta = eta
        self.alpha = alpha

        update_map = {
            "standard": lambda: std_update(self.eta),
            "momentum": lambda: momentum_update(self.eta, self.alpha)
        }

        loss_metric_map = {
            "mse": mse,
            "mee": mee
        }

        if isinstance(update, str):
            try:
                self.update = update_map[update.lower()]()
                print(f"Update algorithm set to: {self.update}")
            except KeyError:
                raise ValueError(f"Unknown Gradient update '{update}'. Available: {list(update_map.keys())}")
        else:
            raise BaseException("Update must be a string")

        if isinstance(loss, str):
            try:
                self.loss = loss_metric_map[loss.lower()]()
                print(f"Loss set to: {self.loss}")
            except KeyError:
                raise ValueError(f"Unknown Loss '{loss}'. Available: {list(loss_metric_map.keys())}")
        else:
            raise BaseException("loss must be a string")

        if isinstance(metric, str):
            try:
                self.metric = loss_metric_map[metric.lower()]()
                print(f"Metric set to: {self.metric}")
            except KeyError:
                raise ValueError(f"Unknown Metric '{metric}'. Available: {list(loss_metric_map.keys())}")
        else:
            raise BaseException("metric must be a string")
            
    def train(self, x, y, batch_size):

        layer_inputs = []
        layer_outputs = []
        
        # Forward prop
        for layer in self.layers:
            layer_inputs.append(x)
            x = layer.forward_pass(x)
            layer_outputs.append(x)
            print("Tisca")
        layer_inputs = np.array(layer_inputs)
        layer_outputs = np.array(layer_outputs)
        delta = self.loss.backward_loss(layer_outputs[-1], y)

        # Backprop
        for inp, layer in zip(reversed(layer_inputs), reversed(self.layers)):
            delta = layer.backward_pass(delta, inp, batch_size)
            print('Tusca')

        # Aggiornamento Pesi
        for layer in self.layers:
            self.update(layer)
            print('Topolino')

layers = [Layer(10, 100, "relu", "standard"), Layer(100, 10, "identity", "standard")]
model = Model(eta=0.05, alpha=0, layers=layers, update="standard", loss="mse", metric="mee")
