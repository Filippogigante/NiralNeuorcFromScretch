import numpy as np
import matplotlib.pyplot as plt

### I data sarebbero i data_train 

data = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TR.csv", delimiter=",", skiprows=1)
data_test = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TS.csv", delimiter=",", skiprows=1)

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

def relu(Z):
    return np.maximum(0, Z)

def drelu(Z):
    return (Z > 0).astype(float)

def tanh(Z):
    return np.tanh(Z)

def init_values(hidden_layer_dim):
    W1 = np.random.randn(hidden_layer_dim, X_train.shape[0]) * np.sqrt(2. / X_train.shape[0])
    b1 = np.zeros((hidden_layer_dim, 1))
    W2 = np.random.randn(4, hidden_layer_dim) * np.sqrt(2. / hidden_layer_dim)
    b2 = np.zeros((4, 1))
    '''W1 = np.random.rand(hidden_layer_dim, X_train.shape[0]) - 0.5
    b1 = np.random.rand(hidden_layer_dim, 1) - 0.5
    W2 = np.random.rand(4, hidden_layer_dim) - 0.5
    b2 = np.random.rand(4, 1) - 0.5'''

    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)

    return W1, b1, vW1, vb1, W2, b2, vW2, vb2

def fp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    return Z1, A1, Z2

def Backprop(Z1, Z2, A1, W2, Y, X):
    m = Y.shape[1]
    dZ2 = (Z2 - Y)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # axis = 1 somma le righe della matrice
    dZ1 = (W2.T).dot(dZ2) * drelu(A1) # con la tanh(1 - A1 ** 2.0)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def aggiorna_parametri(W1, W2, b1, b2, dW1, dW2, vW1, vW2, db1, db2, vb1, vb2, eta, alpha):
    vW1 = alpha * vW1 + (1 - alpha) * dW1
    vb1 = alpha * vb1 + (1 - alpha) * db1
    vW2 = alpha * vW2 + (1 - alpha) * dW2
    vb2 = alpha * vb2 + (1 - alpha) * db2


    W1 = W1 - eta * vW1
    b1 = b1 - eta * vb1
    W2 = W2 - eta * vW2
    b2 = b2 - eta * vb2

    return W1, b1, W2, b2, vW1, vb1, vW2, vb2

def mee(Z, Y):
    return np.mean(np.sqrt(np.sum(np.power(Z - Y, 2), axis=0, keepdims=True)))

def mse(Z, Y):
    return np.mean(np.sum(np.power(Z - Y, 2), axis=0, keepdims=True))

def test(X, Y, W1, W2, b1, b2):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    return mee(Z2, Y)

def scivolo(X, Y, hidden_layer_dim, eta, alpha, batch_size, epochs):
    m = X.shape[1]
    batch_size = int(batch_size)
    W1, b1, vW1, vb1, W2, b2, vW2, vb2 = init_values(hidden_layer_dim)
    err = []

    X_loc = X_train.copy()
    Y_loc = Y_train.copy()

    for i in range(epochs+1):
            
        for k in range(0, m, batch_size):
            X_batch = X_loc[:, k : k + batch_size]
            Y_batch = Y_loc[:, k : k + batch_size]

            Z1, A1, Z2 = fp(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = Backprop(Z1, Z2, A1, W2, Y_batch, X_batch)
            W1, b1, W2, b2, vW1, vb1, vW2, vb2 = aggiorna_parametri(W1, W2, b1, b2, dW1, dW2, vW1, vW2, db1, db2, vb1, vb2, eta, alpha)

        if (i % 50 == 0):
            #Shuffling congiunto di X_train e Y_train
            indices = np.arange(X.shape[1])
            np.random.shuffle(indices)
            X_loc = X_loc[:, indices]
            Y_loc = Y_loc[:, indices]

            Z2_clean = fp(W1, b1, W2, b2, X)

            e = mee(Z2_clean[2], Y)
            err.append(e)
            print(f"Iterazione: {i}/{epochs}")
            print("Errore: ", e)

    err = np.array(err)
    n_parameters = W1.size + b1.size + W2.size + b2.size
    n_neurons = hidden_layer_dim + 4
    return n_neurons, n_parameters, err, test(X_test, Y_test, W1, W2, b1, b2),

epochs = 2000
batch_size = 0.35 * X_train.shape[1]
r = scivolo(X_train, Y_train, hidden_layer_dim=32, eta=0.001, alpha=0.99, batch_size=batch_size, epochs=epochs)
print("Test result:", r[3])
print(f"Number of neurons: {r[0]};", f"Number of parameters: {r[1]}.")
plt.figure('Errore')
plt.plot(r[2], c='red', marker='.', linestyle='-')
plt.xlabel('nx50 Epochs')
plt.ylabel('Errore')
plt.grid()
plt.show()