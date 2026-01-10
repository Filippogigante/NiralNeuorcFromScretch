import numpy as np
from Model import *
from Error_plots import *

data = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TR.csv", delimiter=",", skiprows=1)
#data_test = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TS.csv", delimiter=",", skiprows=1)

m, n = data.shape
np.random.shuffle(data)

data_train = data[:400].T
X_train = data_train[1:13]  # data_train è 500 x 17 non trasposta, la prima colonna è il pattern id, le ultime quattro colonne sono i labels
X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
#X_train = (X_train - X_mean) / (X_std + 1e-8)
Y_train = data_train[13:17]

data_test = data[400:].T
X_test = data_test[1:13]
#X_test = (X_test - X_mean) / (X_std + 1e-8)
Y_test = data_test[13:17]

'''data_test = data_test.T
print(data_test.shape)
X_test = data_test[1:13]
X_test = (X_test - X_mean) / (X_std + 1e-8)'''  

layers = [Layer(X_train.shape[0], 32, "relu", "he"), Layer(32, 12, "relu", "he"), Layer(12, 4, "identity", "he")]
model = Model(eta=0.001, alpha=0.99, layers=layers, update="momentum", loss="mse", metric="mee")
#batch_size = int(X_train.shape[1] / 10)
model.fit(500, X_train, Y_train, X_test, Y_test, batch_size=6)