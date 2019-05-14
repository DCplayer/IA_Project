
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import numpy as np

from red import RR

width = 400
height = 400
center = height//2
white = (255, 255, 255)


def save():
    filename = "image.bmp"
    image1.save(filename)


def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2], fill="black",width=5)


root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

# do the Tkinter canvas drawings (visible)
# cv.create_line([0, center, width, center], fill='green')

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

# do the PIL image/draw (in memory) drawings
# draw.line([0, center, width, center], green)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
# filename = "my_drawing.png"
# image1.save(filename)

button = Button(text="save", command=save)
button.pack()
root.mainloop()


img = PIL.Image.open("image.bmp")
img = img.resize((28, 28), PIL.Image.ANTIALIAS)
img.save('image.bmp')

content = PIL.Image.open("image.bmp").convert("I")
array_content = np.array(content);
#Array_Content es un array numerico que tiene los valores de los colores al reves

for col in array_content:
    for row in col:
        row = (255 - row) / 255.0

input = array_content.reshape(784, 1)
red = RR([784, 189, 91, 10])



red.descensograd()

#Se usara la formula de
#       Nh = (Ns)/(alpha * (Ni + No)), donde

#       Nh      =   cantidad de neuronas en las hidden layers
#       alpha   =   Un numero arbitrario, usualmente entre 2 a 10
#       Ni      =   Numero de neuronas input
#       No      =   Numero de neuronas output
#       Ns      =   Numero de neuronas del test

#Siendo los valores de estos

#       Nh      = [364, 243, 189, 146. 121. 104, 91, 80, 73]
#       alpha   = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#       Ni      = 784
#       No      = 10
#       Ns      = 577672


