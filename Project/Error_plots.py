import numpy as np
import matplotlib.pyplot as plt

class error_plot:

    def plot(self, epochs, err):
        raise NotImplementedError
    
class default_plot(error_plot):
    
    def plot(self, epochs, err):
        plt.figure('Error plot')
        plt.plot(err, c='red', marker='.', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.show()
