import numpy as np

class Initializer_weights:
    def init_weights(self):
        raise NotImplementedError("Devi implementare questo metodo nella sottoclasse")

class std_initializer(Initializer_weights):
    def __init__(self, dim_output, dim_input):
        # Salviamo le dimensioni come attributi dell'istanza
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.W = None
        self.b = None
        self.vW = None
        self.vb = None
    
    # Non serve passare di nuovo le dimensioni, le abbiamo già in self
    def init_weights(self):
        # Inizializzazione standard: Random tra -0.5 e 0.5
        self.W = np.random.rand(self.dim_output, self.dim_input) - 0.5
        self.b = np.random.rand(self.dim_output, 1) - 0.5
        
        # Inizializzazione velocity per momentum (zeri)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        
        return self.W, self.b, self.vW, self.vb