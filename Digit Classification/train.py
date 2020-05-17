import load_datasets
from neural_many_hidden_layer import L_layer_model
from neural_one_hidden_layer import nn_model
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from keras_model import CNN_model
import time

start = time.time()

def prepareData(use_CNN = False):
    X_data , y_data = load_datasets.X_data , load_datasets.y_data

    X_shuffle , y_shuffle = [] , []

    for i in range(X_data.shape[1]):
        X_shuffle.append(X_data[: , i])
        y_shuffle.append(y_data[: , i])

    X_shuffle , y_shuffle = shuffle(X_shuffle , y_shuffle)

    X_shuffle = np.array(X_shuffle).T
    y_shuffle = np.array(y_shuffle).T

    print(X_shuffle.shape)
    print(y_shuffle.shape)

    X_train , X_test , y_train , y_test = X_shuffle[: , 0 : 28560] , X_shuffle[: , 28560 : -1] , \
                                          y_shuffle[: , 0 : 28560] , y_shuffle[: , 28560 : -1]
    if use_CNN:
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T
        X_train = (X_train).reshape(X_train.shape[0], 28, 28, 1).astype('float32')
        X_test = (X_test).reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    return X_train , X_test , y_train , y_test

def training(X_train , X_test , y_train , y_test , use_method):
    if use_method == "one_hidden_layer":
        parameters = nn_model(X_train, y_train , 24 , num_iterations=10000, print_cost=True)
    elif use_method == "many_hidden_layer":
        layers_dims = [28 * 28 , 60 , 60 , 60 , 62]
        parameters = L_layer_model(X_train, y_train, layers_dims , learning_rate=0.0075, num_iterations=10000,
                                   print_cost=True)
    elif use_method == "cnn":
        CNN_model(X_train, y_train, X_test, y_test, 50 , 200 , 62)

X_train , X_test , y_train , y_test = prepareData(use_CNN=True)
end = time.time()
print(str(-start + end))
training(X_train , X_test , y_train , y_test , "cnn")
