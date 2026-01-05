import numpy as np
import pandas as pd

data = pd.read_csv(r"\Users\nicol\Desktop\Universita\ML\digit-recognizer\train.csv")

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


def init_values(first_layer_dim, second_layer_dim):
    W0 = np.random.rand(first_layer_dim, 784) - 0.5
    b0 = np.random.rand(first_layer_dim, 1) - 0.5
    W1 = np.random.rand(second_layer_dim, first_layer_dim) - 0.5
    b1 = np.random.rand(second_layer_dim, 1) - 0.5
    W2 = np.random.rand(10, second_layer_dim) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W0, b0, W1, b1, W2, b2

def fp(W0, b0, W1, b1, W2, b2, X):
    Z0 = W0.dot(X) + b0
    A0 = relu(Z0)
    Z1 = W1.dot(A0) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z0, A0,  Z1, A1, Z2, A2

# Trasformare il vettore Y, che sono numeri in una matrice in cui ogni numero è encodato one hot.
def encode(Y):
    encode_Y = np.zeros((Y.size, 10))
    encode_Y[np.arange(Y.size), Y] = 1
    return encode_Y.T

def Backprop(Z0, Z1, A0, A1, A2, W1, W2, Y, X):  ### Mental Reset sulla Backprop.
    m = Y.size
    Y = encode(Y)
    dZ2 = A2 - Y   ## Abbiamo usato come funzione di Loss la Cross Entropy Sum(-Y * logA2)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # axis = 1 somma le righe della matrice, keepdims=True mantiene la shape corretta (10, 1)
    dZ1 = (W2.T).dot(dZ2) * (drelu(Z1))
    dW1 = (1 / m) * dZ1.dot(A0.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    dZ0 = (W1.T).dot(dZ1) * drelu(Z0)
    dW0 = (1 / m) * dZ0.dot(X.T)
    db0 = (1 / m) * np.sum(dZ0, axis=1, keepdims=True)
    
    return dW0, db0, dW1, db1, dW2, db2

def aggiorna_parametri(W0, W1, W2, b0, b1, b2, dW0, dW1, dW2, db0, db1, db2, eta):
    W0 = W0 - eta * dW0
    b0 = b0 - eta * db0
    W1 = W1 - eta * dW1
    b1 = b1 - eta * db1
    W2 = W2 - eta * dW2
    b2 = b2 - eta * db2
    return W0, b0, W1, b1, W2, b2

def accuracy(A2, Y):
    m = Y.size
    k = np.argmax(A2, axis=0)
    a = np.array(Y == k)
    return np.sum(a) / Y.size

def test(X, Y, W0, b0, W1, b1, W2, b2):
    Z0 = W0.dot(X) + b0
    A0 = relu(Z0)
    Z1 = W1.dot(A0) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return accuracy(A2, Y)

def scivolo(X, Y, first_layer_dim, second_layer_dim, eta, iterazioni):
    W0, b0, W1, b1, W2, b2 = init_values(first_layer_dim, second_layer_dim)
    
    for i in range(iterazioni+1):

        Z0, A0, Z1, A1, Z2, A2 = fp(W0, b0, W1, b1, W2, b2, X_train)
        dW0, db0, dW1, db1, dW2, db2 = Backprop(Z0, Z1, A0, A1, A2, W1, W2, Y_train, X_train)
        W0, b0, W1, b1, W2, b2 = aggiorna_parametri(W0, W1, W2, b0, b1, b2, dW0, dW1, dW2, db0, db1, db2, eta)
        if (i % 50 == 0):

            ## Shuffling congiunto di X_train e Y_train
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            print(f"Iterazione: {i}/{iterazioni}")
            print("Accuratezza: ", accuracy(A2, Y_train))
    n_parameters = X.shape[0] * first_layer_dim + first_layer_dim * second_layer_dim + second_layer_dim * 10
    n_neurons = first_layer_dim + second_layer_dim + 10

    return test(X_test, Y_test, W0, b0, W1, b1, W2, b2), n_neurons, n_parameters


r = scivolo(X_train, Y_train, first_layer_dim=25, second_layer_dim=10, eta=0.05, iterazioni=500)
print("Test result:", r[0])
print(f"Number of neurons: {r[1]};", f"Number of parameters: {r[2]}.")