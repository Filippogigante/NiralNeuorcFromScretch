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

    def forward_pass_model(self, x):

        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
    
    def train(self, x, y, batch_size):

        layer_inputs = []
        layer_outputs = []
        # Forward prop
        for layer in self.layers:
            layer_inputs.append(x)
            x = layer.forward_pass(x)
            layer_outputs.append(x)
            
        layer_inputs = np.array(layer_inputs)
        layer_outputs = np.array(layer_outputs)

        # Backprop
        delta = self.loss.backward_loss(layer_outputs[-1], y)
        for inp, layer in zip(reversed(layer_inputs), reversed(self.layers)):
            delta = layer.backward_pass(delta, inp, batch_size)
            

        # Aggiornamento Pesi
        for layer in self.layers:
            self.update.update(layer)
            

    def fit(self, epochs, x, y, batch_size):
        n_train = x.shape[1]
        X_loc = x.copy()
        Y_loc = y.copy()
        err = []

        for epoch in range(epochs+1):
            for k in range(0, n_train, batch_size):
                X_batch = X_loc[:, k : k + batch_size]
                Y_batch = Y_loc[:, k : k + batch_size]

                self.train(X_batch, Y_batch, batch_size)
            if (epoch % np.floor(epochs/10) == 0):
                #Shuffling congiunto di X_train e Y_train
                indices = np.arange(x.shape[1])
                np.random.shuffle(indices)
                X_loc = X_loc[:, indices]
                Y_loc = Y_loc[:, indices]

                o = self.forward_pass_model(x)

                e = self.metric.forward_loss(o, y)
                err.append(e)
                print(f"Epoch: {epoch}/{epochs}")
                print("Errore: ", e)
            err = np.array(err)
            return err
    
    def summary(self):
        neurons = 0
        n_params = 0
        for layer in self.layers:
            neurons += layer.dim_input
            n_params += layer.get_weights().size + layer.get_bias().size
        
        print(f"--------------------------------------------------------------------")
        print(f"Number of neurons: {neurons};", f"Number of parameters: {n_params}.")
        print(f"Number of Mbyte: {(n_params * 8) / 1024}")
        print(f"Learning rate: {self.eta}", f"Momentum coefficient: {self.alpha}")
        print(f"--------------------------------------------------------------------")
