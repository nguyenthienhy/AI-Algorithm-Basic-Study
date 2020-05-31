from PIL import Image
import os
import numpy as np
import constant
import process_image

assert (len(constant.defines) == 36)

def readImage(path, convert_to_gray):
    if not convert_to_gray:
        image = Image.open(path)
        array = np.array(image.resize((28, 28)), dtype=np.float32)
        #array = process_image.convert_to_black_background(array)
        return array.reshape(28 * 28 * 3, 1) / 255.0
    else:
        image = Image.open(path).convert('L')
        array = np.array(image.resize((28, 28)), dtype=np.float32)
        #array = process_image.convert_to_black_background(array)
        return array.reshape(28 * 28, 1) / 255.0


def readDataForOneLabel(X_data , y_data , path, label):
    for entry in os.listdir(path):
        if os.path.isfile((os.path.join(path, entry))):
            X_data.append(readImage(path + '/' + entry, True))
            for index, label_digit in enumerate(constant.defines):
                if label == label_digit:
                    temp_label = np.zeros((1, 36))
                    temp_label[0][index] = 1
                    y_data.append(temp_label[0])

def get_list_images_for_one_label(L , y , path , label):
    for entry in os.listdir(path):
        if os.path.isfile((os.path.join(path, entry))):
            L.append(path + '/' + entry)
            for index, label_digit in enumerate(constant.defines):
                if label == label_digit:
                    y.append(index)

def get_all_images_sub_dir(L , y , basepath , label):
    for entry in os.listdir(basepath):
        basepath_sub = basepath
        if os.path.isdir((os.path.join(basepath_sub, entry))):
            basepath_sub = basepath + '/' + entry
            get_list_images_for_one_label(L , y , basepath_sub , label)
            readSubDir(L , y , basepath_sub, label)

def readSubDir(X_data , y_data , basepath, label):
    for entry in os.listdir(basepath):
        basepath_sub = basepath
        if os.path.isdir((os.path.join(basepath_sub, entry))):
            basepath_sub = basepath + '/' + entry
            readDataForOneLabel(basepath_sub, label)
            readSubDir(X_data , y_data , basepath_sub, label)

def readImages(path):
    L = []
    y = []
    for l in range(len(constant.defines)):
        get_all_images_sub_dir(L , y , (path + '/' + str(constant.defines[l])), constant.defines[l])
        get_list_images_for_one_label(L , y , (path + '/' + str(constant.defines[l])), constant.defines[l])
    return L , y

def readData(path):
    X_data = []
    y_data = []
    for l in range(len(constant.defines)):
        readSubDir(X_data , y_data , (path + '/' + str(constant.defines[l])), constant.defines[l])
        readDataForOneLabel(X_data , y_data , (path + '/' + str(constant.defines[l])), constant.defines[l])
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    return X_data , y_data

def get_data(Data_dir):
    X_data , y_data = readData(Data_dir)
    X_data = X_data.reshape(X_data.shape[0], -1).T
    y_data = y_data.T
    return X_data , y_data

