from matplotlib import pyplot as plt
from PIL import Image
import cnn_model
import os

def readTest(path):
    List_Images = []
    for entry in os.listdir(path):
        if os.path.isfile((os.path.join(path, entry))):
            List_Images.append(path + '/' + entry)
    return List_Images

def show_results(model , List_Images , columns , rows):
    fig = plt.figure(figsize=(8 , 8))
    for i in range(1, columns * rows + 1):
        img = Image.open(List_Images[i])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.title("Predicted: " + str(cnn_model.predict(model , List_Images[i])))
    plt.show()


