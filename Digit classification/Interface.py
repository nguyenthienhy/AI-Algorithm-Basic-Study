from tkinter import *
from tkinter.ttk import *
import PIL.ImageGrab
from PIL import Image
from cnn_model import *
import numpy as np

# load model
model = load_model()

# create global variables 
operator = "Predicted Number: "
operator2 = ""

# create function to clear canvas and text
def Clear():
    cv.delete("all")
    global operator2
    text_input.set(operator2)

# create function to predict and display predicted number
def Predict():
    file = 'image.jpg'
    if file:
        # save the canvas in jpg format
        x = root.winfo_rootx() + cv.winfo_x() + 42
        y = root.winfo_rooty() + cv.winfo_y() + 35
        x1 = x + cv.winfo_width()
        y1 = y + cv.winfo_height()
        PIL.ImageGrab.grab().crop((x,y,x1,y1)).save(file)
        y_pred = predict(model , file)
        text_input.set("Predicted : " + str(y_pred))

# create function to draw on canvas
def paint(event):
    old_x = event.x
    old_y = event.y        
        
    cv.create_line(old_x, old_y, event.x, event.y,
                               width=8, fill="black",
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)

# all interface elements must be between Tk() and mainloop()
root = Tk()

#create string variable
text_input = StringVar()

#create field to display text
textdisplay = Entry(root, 
               textvariable = text_input,  
               justify = 'center')

# create predict and clear buttons
btn1 = Button(root, text = "Predict", command = lambda:Predict())
btn2 = Button(root, text = "Clear", command = lambda:Clear())

#create canvas to draw on
cv = Canvas(root,width=200,height=200,bg="white",)

#using left mouse button to draw
cv.bind('<B1-Motion>', paint)

#organise the elements
cv.grid(row = 0, column = 0)
textdisplay.grid(row = 0, column = 1)
btn1.grid(row = 1, column = 0)
btn2.grid(row = 1, column = 1)

#this 2 lines for expand the interface
root.rowconfigure(0, weight=2)
root.columnconfigure(1, weight=2)

root.mainloop()