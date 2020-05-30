from PIL import Image
import os
import numpy as np

defines = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "a_u", "b_u", "c_u", "d_u",
           "e_u", "f_u", "g_u", "h_u",
           "i_u", "j_u", "k_u", "l_u",
           "m_u", "n_u", "o_u", "p_u",
           "q_u", "r_u", "s_u", "t_u",
           "u_u", "v_u", "w_u", "x_u",
           "y_u", "z_u"]

assert (len(defines) == 36)

X_data = []
y_data = []

def readImage(path, convert_to_gray):
    if not convert_to_gray:
        image = Image.open(path)
        array = np.array(image.resize((28, 28)), dtype=np.float32)
        return array.reshape(28 * 28 * 3 , 1) / 255.0
    else:
        image = Image.open(path).convert('L')
        array = np.array(image.resize((28, 28)), dtype=np.float32)
        '''
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] >= 127:
                    array[i][j] = 0
                else:
                    array[i][j] = 255
        '''
        return array.reshape(28 * 28, 1) / 255.0


def readDataForOneLabel(path, label):
    for entry in os.listdir(path):
        if os.path.isfile((os.path.join(path, entry))):
            X_data.append(readImage(path + '/' + entry, True))
            for index, label_digit in enumerate(defines):
                if label == label_digit:
                    temp_label = np.zeros((1, 36))
                    temp_label[0][index] = 1
                    y_data.append(temp_label[0])

def readSubDir(basepath, label):
    for entry in os.listdir(basepath):
        basepath_sub = basepath
        if os.path.isdir((os.path.join(basepath_sub, entry))):
            basepath_sub = basepath + '/' + entry
            readDataForOneLabel(basepath_sub, label)
            readSubDir(basepath_sub, label)

def readData(path):
    for l in range(len(defines)):
        readSubDir((path + '/' + str(defines[l])), defines[l])
        readDataForOneLabel((path + '/' + str(defines[l])), defines[l])


readData("Data")

Original_Data = X_data

X_data = np.asarray(X_data)
y_data = np.asarray(y_data)

X_data = X_data.reshape(X_data.shape[0], -1).T
y_data = y_data.T
