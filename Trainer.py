import numpy as np
from sklearn.model_selection import train_test_split
from red import RR

print("Loading data...")
circles = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/circle.npy")
egg = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/egg.npy")
face = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/face.npy")
house = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/house.npy")
mickey = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/mickey.npy")
question = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/question.npy")
sad = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/sad.npy")
square = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/square.npy")
tree = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/tree.npy")
triangle = np.load("/home/diego/Desktop/NovenoSemestre/Inteligencia/data/triangle.npy")

circles_y  = np.array([1,0,0,0,0,0,0,0,0,0]).reshape(10, 1)
egg_y      = np.array([0,1,0,0,0,0,0,0,0,0]).reshape(10, 1)
face_y     = np.array([0,0,1,0,0,0,0,0,0,0]).reshape(10, 1)
house_y    = np.array([0,0,0,1,0,0,0,0,0,0]).reshape(10, 1)
mickey_y   = np.array([0,0,0,0,1,0,0,0,0,0]).reshape(10, 1)
question_y = np.array([0,0,0,0,0,1,0,0,0,0]).reshape(10, 1)
sad_y      = np.array([0,0,0,0,0,0,1,0,0,0]).reshape(10, 1)
square_y   = np.array([0,0,0,0,0,0,0,1,0,0]).reshape(10, 1)
tree_y     = np.array([0,0,0,0,0,0,0,0,1,0]).reshape(10, 1)
triangle_y = np.array([0,0,0,0,0,0,0,0,0,1]).reshape(10, 1)

data = []
i = 0
amount = 8000

while i < amount and i < len(circles):
    data.append(((circles[i] / 255).reshape(784, 1), circles_y))
    i += 1
i = 0
while i < amount and i < len(egg):
    data.append(((egg[i] / 255.0).reshape(784, 1), egg_y))
    i += 1
i = 0
while i < amount and i < len(face):
    data.append(((face[i] / 255.0).reshape(784, 1), face_y))
    i += 1
i = 0
while i < amount and i < len(house):
    data.append(((house[i] / 255.0).reshape(784, 1), house_y))
    i += 1
i = 0
while i < amount and i < len(mickey):
    data.append(((mickey[i] / 255.0).reshape(784, 1), mickey_y))
    i += 1
i = 0
while i < amount and i < len(question):
    data.append(((question[i] / 255.0).reshape(784, 1), question_y))
    i += 1
i = 0
while i < amount and i < len(sad):
    data.append(((sad[i] / 255.0).reshape(784, 1), sad_y))
    i += 1
i = 0
while i < amount and i < len(square):
    data.append(((square[i] / 255.0).reshape(784, 1), square_y))
    i += 1
i = 0
while i < amount and i < len(tree):
    data.append(((tree[i] / 255.0).reshape(784, 1), tree_y))
    i += 1
i = 0
while i < amount and i < len(triangle):
    data.append(((triangle[i] / 255.0).reshape(784, 1), triangle_y))
    i += 1
print("Data loaded.")


###Training
print("Starting training...")
train, otherT = train_test_split(data, test_size=0.30, random_state=19)
test, cross = train_test_split(otherT, test_size=0.5, random_state=55)
red = RR([784, 189, 10])
red.descensograd(train, test, cross, 3, 60)