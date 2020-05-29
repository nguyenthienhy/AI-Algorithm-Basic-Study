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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint

start = time.time()

def load_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))
    model.load_weights("weights.best.hdf5")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def prepareData(use_CNN=False):
    X_data, y_data = load_datasets.X_data, load_datasets.y_data

    X_shuffle, y_shuffle = [], []

    for i in range(X_data.shape[1]):
        X_shuffle.append(X_data[:, i])
        y_shuffle.append(y_data[:, i])

    X_shuffle, y_shuffle = shuffle(X_shuffle, y_shuffle)

    X_shuffle = np.array(X_shuffle).T
    y_shuffle = np.array(y_shuffle).T

    print(X_shuffle.shape)
    print(y_shuffle.shape)

    # get 80 percent train
    num_train = int(X_shuffle.shape[1] * 0.8)
    print("Number of trainings : " + str(num_train))

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

    return X_train, X_test, y_train, y_test


def training(X_train, X_test, y_train, y_test, use_method):
    if use_method == "one_hidden_layer":
        parameters = nn_model(X_train, y_train, 24,
                              num_iterations=10000, print_cost=True)
    elif use_method == "many_hidden_layer":
        layers_dims = [28 * 28, 60, 60, 60, 62]
        parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=0.0075, num_iterations=10000,
                                   print_cost=True)
    elif use_method == "cnn":
        CNN_model(X_train, y_train, X_test, y_test, 50 , 200 , 36)

def predict(model , path):
    # show image of test
    image = Image.open(path).convert('L')
    image.show()
    array = np.array(image.resize((28, 28)), dtype=np.float32)
    array = array.reshape(28 * 28 , 1) / 255.0
    array = np.array([array])
    array = array.reshape(1 , 28 , 28 , 1).astype('float32')
    return model.predict_classes(array)

#X_train, X_test, y_train, y_test = prepareData(use_CNN=True)
#end = time.time()
#print("Load data take : " + str(end - start) + "s")
#training(X_train, X_test, y_train, y_test, "cnn")
model = load_model()
print(predict(model , "test.png")[0])
