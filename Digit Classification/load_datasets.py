from PIL import Image
import os
import numpy as np

X_data = []
y_data = []

def readImage(path):
    image = Image.open(path)
    return np.array((image).resize((20 , 20)))

def readDataForOneLabel(path , label , maxTake):
    count = 0
    for entry in os.listdir(path):
        if os.path.isfile((os.path.join(path, entry))):
            count += 1
            X_data.append(readImage(path + '/' + entry))
            if label == 0:
                temp_label = np.zeros((1 , 10))
                temp_label[0][0] = 1
                y_data.append(temp_label[0])
            else:
                temp_label = np.zeros((1 , 10))
                temp_label[0][label] = 1
                y_data.append(temp_label[0])
            if count == maxTake:
                break

def readDataAll(path , maxTake):
    for l in range(10):
        readDataForOneLabel((path + '/' + str(l)) , l , maxTake)

readDataAll("Data" , 4200)

X_data = np.asarray(X_data)
y_data = np.asarray(y_data)

X_data = X_data.reshape(X_data.shape[0] , -1).T
y_data = y_data.T

