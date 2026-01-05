import numpy as np
import pandas as pd

## Abbiamo usato come funzione di Loss la Cross Entropy Sum(-Y * logA2)

data = pd.read_csv(r"\Users\nicol\Desktop\Universita\ML\digit-recognizer\train.csv")
#data = pd.read_csv(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TR.csv") è un bel casino sono quattro target

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
X_test = data_dev[1:n] / 255.0 #Normaliziamo i valori altrimenti esplode tutto
Y_test = data_dev[0]

data_train = data[1000:m].T
X_train = data_train[1:n] / 255.0
Y_train = data_train[0]


def relu(Z):
    return np.maximum(0, Z)

def drelu(Z):
    return Z > 0

def softmax(Z):
    return np.exp(Z) / (np.sum(np.exp(Z), axis=0, keepdims=True))


def init_values(hidden_layer_dim):
    W1 = np.random.rand(hidden_layer_dim, 784) - 0.5
    b1 = np.random.rand(hidden_layer_dim, 1) - 0.5
    W2 = np.random.rand(10, hidden_layer_dim) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def fp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#Bisogna trasformare il vettore Y, che sono numeri in una matrice in cui ogni numero è encodato ammodo.
def encode(Y):
    encode_Y = np.zeros((Y.size, 10))
    encode_Y[np.arange(Y.size), Y] = 1
    return encode_Y.T

def Backprop(Z1, Z2, A1, A2, W1, W2, b1, b2, Y, X):  ### Mental Reset sulla Backprop.
    m = Y.size
    Y = encode(Y)
    dZ2 = A2 - Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # axis = 1 somma le righe della matrice
    dZ1 = (W2.T).dot(dZ2) * (drelu(Z1))
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def aggiorna_parametri(W1, W2, b1, b2, dW1, dW2, db1, db2, eta):
    W1 = W1 - eta * dW1
    b1 = b1 - eta * db1
    W2 = W2 - eta * dW2
    b2 = b2 - eta * db2
    return W1, b1, W2, b2

def accuracy(A2, Y):
    m = Y.size
    k = np.argmax(A2, axis=0)
    a = np.array(Y == k)
    return np.sum(a) / Y.size

def test(X, Y, W1, W2, b1, b2):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return accuracy(A2, Y)

def scivolo(X, Y, hidden_layer_dim, eta, iterazioni):
    W1, b1, W2, b2 = init_values(hidden_layer_dim)
    
    for i in range(iterazioni+1):

        Z1, A1, Z2, A2 = fp(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = Backprop(Z1, Z2, A1, A2, W1, W2, b1, b2, Y_train, X_train)
        W1, b1, W2, b2 = aggiorna_parametri(W1, W2, b1, b2, dW1, dW2, db1, db2, eta)
        if (i % 50 == 0):

            ## Shuffling congiunto di X_train e Y_train
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            print(f"Iterazione: {i}/{iterazioni}")
            print("Accuratezza: ", accuracy(A2, Y_train))

    n_parameters = X.shape[0] * hidden_layer_dim + hidden_layer_dim * 10
    n_neurons = hidden_layer_dim + 10
    return test(X_test, Y_test, W1, W2, b1, b2), n_neurons, n_parameters


r = scivolo(X_train, Y_train, hidden_layer_dim=10, eta=0.1, iterazioni=600)
print("Test result:", r[0])
print(f"Number of neurons: {r[1]};", f"Number of parameters: {r[2]}.")