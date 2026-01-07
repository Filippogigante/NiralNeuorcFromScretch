import numpy as np
import matplotlib.pyplot as plt

class error_plot:

    def plot(self, epochs, err):
        raise NotImplementedError
    
class default_plot(error_plot):
    
    def plot(self, epochs, err):

        step = int(np.floor(epochs/10))
        x_epochs = np.array([i for i in range(0, epochs+1, step)])
        plt.figure('Error plot')
        plt.plot(x_epochs, err, c='red', marker='.', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.show()
