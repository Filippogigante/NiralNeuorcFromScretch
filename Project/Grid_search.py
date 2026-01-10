import numpy as np
import shutil
import os 
import pickle
from Model import *
from Error_plots import *
import itertools

folder_path = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights"
original_file = "best_weights.pkl"
new_file = "global_best_weights.pkl"
old_path = os.path.join(folder_path, original_file)
new_path = os.path.join(folder_path, new_file)

data = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TR.csv", delimiter=",", skiprows=1)
#data_test = np.loadtxt(r"\Users\nicol\Desktop\Universita\ML\cup_data\ML-CUP25-TS.csv", delimiter=",", skiprows=1)

m, n = data.shape
np.random.shuffle(data)

data_train = data[:400].T
X_train = data_train[1:13]  # data_train è 500 x 17 non trasposta, la prima colonna è il pattern id, le ultime quattro colonne sono i labels
X_mean = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
X_train = (X_train - X_mean) / (X_std + 1e-8)
Y_train = data_train[13:17]

data_val = data[400:].T
X_val = data_val[1:13]
X_val = (X_val - X_mean) / (X_std + 1e-8)
Y_val = data_val[13:17]

'''data_test = data_test.T
print(data_test.shape)
X_test = data_test[1:13]
X_test = (X_test - X_mean) / (X_std + 1e-8)'''  

param_grid = {"eta": [0.01],
                "alpha": [0.99],
                "batch_size": [16, 32],
                "update": ["momentum"],
                "initializer": ["he"],
                "activation": ["relu", "tanh"],
                "hidden_architecture": [
                    (32,),          # 1 hidden layer da 32
                    (64,),          # 1 hidden layer da 64
                    (32, 32),       # 2 hidden layers da 32
                    (32, 12),       # 2 hidden layers: primo 64, secondo 32 (a imbuto)
                    (64, 32, 12)
                ],
                "epochs": [200]
                 }

keys = param_grid.keys()
values = param_grid.values()
combinations = list(itertools.product(*values))

print(f"Inizio Grid Search su {len(combinations)} combinazioni...")

best_params = None
global_best_val_loss = float('inf')

for combo in combinations:
    params = dict(zip(keys, combo))
    
    current_arch = params["hidden_architecture"]
    n_layers = len(current_arch)
    print(f"Hidden architecture: {current_arch}, number of hidden Layers: {n_layers}")

    layers =[]
    full_structure = [X_train.shape[0]] + list(current_arch) + [4] # Attacchiamo alla lista con la dimensione degli hidden layers la dimensione del primo e dell'ultimo
    for i in range(len(full_structure) - 1):
        
        if i == len(full_structure) - 2:
            act_func = "identity" 
        else:
            act_func = params["activation"]
            
        layer = Layer(
            dim_input=full_structure[i], 
            dim_output=full_structure[i+1], 
            activation=act_func, 
            initializer=params["initializer"]
        )
        layers.append(layer)

    model = Model(eta=params["eta"], alpha=params["alpha"], layers=layers, update=params["update"], loss="mse", metric="mee")
    batch_size = int(X_train.shape[1] / 10)
    best_val_loss = model.fit(params["epochs"], X_train, Y_train, X_val, Y_val, params["batch_size"])

    if best_val_loss < global_best_val_loss:
        global_best_val_loss = best_val_loss
        best_params = params
        shutil.copy(old_path, new_path)
        print(f"--> Nuova configurazione migliore trovata! Loss: {global_best_val_loss:.2f}")

print("\n========================================")
print(f"Grid Search Completata.")
print(f"Migliore Configurazione: {best_params}")
print(f"Migliore Validation Loss: {global_best_val_loss}")
print("========================================")

    #print(f"Validation Error: {model.evaluate(X_val, Y_val)}")

