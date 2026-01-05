import numpy as np
import matplotlib.pyplot as plt

### I data sarebbero i data_train 

data = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TR.csv", delimiter=",", skiprows=1)
data_test = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TS.csv", delimiter=",", skiprows=1)

m, n = data.shape
np.random.shuffle(data)

data_train = data[0:400].T
X_train = data_train[1:13]  # data_train è 500 x 17, la prima colonna è il pattern id, le ultime quattro colonne sono i labels
X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
X_train = (X_train - X_mean) / (X_std + 1e-8)
Y_train = data_train[13:17]
Y_mean = np.mean(Y_train, axis=1, keepdims=True)
Y_std = np.std(Y_train, axis=1, keepdims=True)
#Y_train = (Y_train - Y_mean) / (Y_std + 1e-8)

data_test = data[400:].T
X_test = data_test[1:13]
X_test = (X_test - X_mean) / (X_std + 1e-8)
Y_test = data_test[13:17]

'''data_test = data_test.T
print(data_test.shape)
X_test = data_test[1:13]
X_test = (X_test - X_mean) / (X_std + 1e-8)'''


def relu(Z):
    return np.maximum(0, Z)

def drelu(Z):
    return Z > 0

def tanh(Z):
    return np.tanh(Z)


def init_values(first_layer_dim, second_layer_dim):
    W1 = np.random.randn(first_layer_dim, X_train.shape[0]) * np.sqrt(2. / X_train.shape[0])
    b1 = np.zeros((first_layer_dim, 1))
    W2 = np.random.randn(second_layer_dim, first_layer_dim) * np.sqrt(2. / first_layer_dim)
    b2 = np.zeros((second_layer_dim, 1))
    W3 = np.random.randn(4, second_layer_dim) * np.sqrt(2. / second_layer_dim)
    b3 = np.zeros((4, 1))
    '''W1 = np.random.rand(hidden_layer_dim, X_train.shape[0]) - 0.5
    b1 = np.random.rand(hidden_layer_dim, 1) - 0.5
    W2 = np.random.rand(4, hidden_layer_dim) - 0.5
    b2 = np.random.rand(4, 1) - 0.5'''

    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    vW3 = np.zeros_like(W3)
    vb3 = np.zeros_like(b3)
    return W1, b1, vW1, vb1, W2, b2, vW2, vb2, W3, b3, vW3, vb3

def fp(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    return Z1, A1, Z2, A2, Z3

def Backprop(Z1, Z2, Z3, A1, A2, W2, W3, Y, X):  ### Mental Reset sulla Backprop.
    
    dZ3 = (Z3 - Y) / np.sqrt((np.sum(np.power(Z3 - Y, 2), axis=1, keepdims=True)))
    dW3 = dZ3.dot(A2.T)
    db3 = np.sum(dZ3, axis=1, keepdims=True) # axis = 1 somma le righe della matrice
    dZ2 = (W3.T).dot(dZ3) * drelu(Z2)#(1 - A2 ** 2.0)
    dW2 = dZ2.dot(A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = (W2.T).dot(dZ2) * drelu(Z1) #(1 - A1 ** 2.0)
    dW1 = dZ1.dot(X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def aggiorna_parametri(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3, vW1, vW2, vW3, vb1, vb2, vb3, eta, alpha):
    vW1 = alpha * vW1 + [1 - alpha] * dW1
    vb1 = alpha * vb1 + [1 - alpha] * db1
    vW2 = alpha * vW2 + [1 - alpha] * dW2
    vb2 = alpha * vb2 + [1 - alpha] * db2
    vW3 = alpha * vW3 + [1 - alpha] * dW3
    vb3 = alpha * vb3 + [1 - alpha] * db3

    W1 = W1 - eta * vW1
    b1 = b1 - eta * vb1
    W2 = W2 - eta * vW2
    b2 = b2 - eta * vb2
    W3 = W3 - eta * vW3
    b3 = b3 - eta * vb3
    return W1, b1, W2, b2, W3, b3, vW1, vb1, vW2, vb2, vW3, vb3

def mee(Z, Y):
    return np.mean(np.sqrt(np.sum(np.power(Z - Y, 2), axis=0, keepdims=True)))

def mse(Z, Y):
    return np.mean(np.sum(np.power(Z - Y, 2), axis=0, keepdims=True))

def test(X, Y, Y_std, Y_mean, W1, W2, W3, b1, b2, b3):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) +b3
    return mee(Z3, Y)

def scivolo(X, Y, first_layer_dim, second_layer_dim, eta, alpha, iterazioni):
    W1, b1, vW1, vb1, W2, b2, vW2, vb2, W3, b3, vW3, vb3 = init_values(first_layer_dim, second_layer_dim)
    err = []
    X_loc = X_train.copy()
    Y_loc = Y_train.copy()
    for i in range(iterazioni+1):

        Z1, A1, Z2, A2, Z3 = fp(W1, b1, W2, b2, W3, b3, X_loc)
        dW1, db1, dW2, db2, dW3, db3 = Backprop(Z1, Z2, Z3, A1, A2, W2, W3, Y_loc, X_loc)
        W1, b1, W2, b2, W3, b3, vW1, vb1, vW2, vb2, vW3, vb3 = aggiorna_parametri(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3, vW1, vW2, vW3, vb1, vb2, vb3, eta, alpha)
        if (i % 50 == 0):
            #Shuffling congiunto di X_train e Y_train
            indices = np.arange(X.shape[1])
            np.random.shuffle(indices)
            X_loc = X_loc[:, indices]
            Y_loc = Y_loc[:, indices]

            Z3_clean = fp(W1, b1, W2, b2, W3, b3, X)

            e = mee(Z3_clean[4], Y)
            err.append(e)
            print(f"Iterazione: {i}/{iterazioni}")
            print("Errore: ", e)
    err = np.array(err)
    n_parameters = W1.size + b1.size + W2.size + b2.size + W3.size + b3.size
    n_neurons = first_layer_dim + second_layer_dim + 4
    return n_neurons, n_parameters, err, test(X_test, Y_test, Y_std, Y_mean, W1, W2, W3, b1, b2, b3),

iterazioni = 2000
r = scivolo(X_train, Y_train, first_layer_dim=64, second_layer_dim=32, eta=0.001, alpha=0.9, iterazioni=iterazioni)
print("Test result:", r[3])
print(f"Number of neurons: {r[0]};", f"Number of parameters: {r[1]}.")
plt.figure('Errore')
plt.plot(r[2], c='red', marker='.', linestyle='-')
plt.xlabel('nx50 iterazioni')
plt.ylabel('Errore')
plt.grid()
plt.show()