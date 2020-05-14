from PIL import Image
import os
import numpy as np

defines = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "a", "a_u", "b", "b_u", "c", "c_u", "d", "d_u",
           "e", "e_u", "f", "f_u", "g", "g_u", "h", "h_u",
           "i", "i_u", "j", "j_u", "k", "k_u", "l", "l_u",
           "m", "m_u", "n", "n_u", "o", "o_u", "p", "p_u",
           "q", "q_u", "r", "r_u", "s", "s_u", "t", "t_u",
           "u", "u_u", "v", "v_u", "w", "w_u", "x", "x_u",
           "y", "y_u", "z", "z_u"]

assert(len(defines) == 62)

X_data = []
y_data = []


def readImage(path, convert_to_gray):
    if convert_to_gray == False:
        image = Image.open(path)
        array = np.array((image).resize((28, 28)), dtype=np.float32)
        return (255 - np.array((image).resize((28, 28)), dtype=np.float32).reshape(28 * 28 * 3, 1)) / 255.0
    else:
        image = Image.open(path).convert('L')
        return (255 - np.array((image).resize((28, 28)), dtype=np.float32).reshape(28 * 28, 1)) / 255.0


def readDataForOneLabel(path, label, maxTake):
    count = 0
    for entry in os.listdir(path):
        if os.path.isfile((os.path.join(path, entry))):
            count += 1
            X_data.append(readImage(path + '/' + entry, True))
            for index, label_digit in enumerate(defines):
                if label == label_digit:
                    temp_label = np.zeros((1, 62))
                    temp_label[0][index] = 1
                    y_data.append(temp_label[0])
            if count == maxTake:
                break


def readDataAll(path, maxTake):
    for l in range(len(defines)):
        readDataForOneLabel(
            (path + '/' + str(defines[l])), defines[l], maxTake)


readDataAll("Data", 400)

Original_Data = X_data

X_data = np.asarray(X_data)
y_data = np.asarray(y_data)

X_data = X_data.reshape(X_data.shape[0], -1).T
y_data = y_data.T
