from neural_many_hidden_layer import L_layer_model
from neural_one_hidden_layer import nn_model
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import numpy as np
from cnn_model import *
import time
import show_result as sr

start = time.time()

def prepareData(use_CNN=False):
    import load_datasets

    X_data, y_data = load_datasets.get_data("Data")

    X_shuffle, y_shuffle = [], []

    for i in range(X_data.shape[1]):
        X_shuffle.append(X_data[:, i])
        y_shuffle.append(y_data[:, i])

    X_shuffle, y_shuffle = shuffle(X_shuffle, y_shuffle)

    X_shuffle = np.array(X_shuffle).T
    y_shuffle = np.array(y_shuffle).T

    # get 80 percent train
    num_train = int(X_shuffle.shape[1] * 0.8)

    X_train, X_test, y_train, y_test = X_shuffle[:, 0: num_train], X_shuffle[:, num_train: -1], \
        y_shuffle[:, 0: num_train], y_shuffle[:, num_train: -1]

    if use_CNN:
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T
        X_train = (X_train).reshape(
            X_train.shape[0], 28, 28, 1).astype('float32')
        X_test = (X_test).reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    return X_train, X_test, y_train , y_test

def create_training_session(X_train, X_test, y_train, y_test, use_method):
    if use_method == "one_hidden_layer":
        parameters = nn_model(X_train, y_train, 24,
                              num_iterations=10000, print_cost=True)
    elif use_method == "many_hidden_layer":
        layers_dims = [28 * 28, 60, 60, 60, 62]
        parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.0075, num_iterations=10000,
                                   print_cost=True)
    elif use_method == "cnn":
        CNN_model(X_train, y_train, X_test, y_test, 60 , 512 , 36)

def train():
    X_train, X_test, y_train, y_test = prepareData(use_CNN=True)
    create_training_session(X_train, X_test, y_train, y_test , "cnn")

model = load_model()
'''
import load_datasets
List_Images , y_true = load_datasets.readImages("Test_full")
y_redict = []
for im in List_Images:
    y_redict.append(predictOutNum(model , im))
print(classification_report(y_true , y_redict))
'''
List_Images = sr.readTest("Test")
List_Images = shuffle(List_Images)
sr.show_results(model , List_Images , 4 , 5)