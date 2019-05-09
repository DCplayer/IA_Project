import matplotlib.pyplot as plt
import numpy as np
import random


class RR(object):
    def __init__(self, tamanios):
        self.capas = len(tamanios)
        self.tamanios = tamanios
        self.biases = []
        self.weight = []
        self.assign_value()
        self.assign_weights()

    def assign_value(self):
        i = 1
        for i in range(len(self.tamanios)):
            random = np.random.randn(i, 1)
            self.biases.append(random)

    def assign_weights(self):
        for x, y in zip(self.tamanios[:-1], self.tamanios[1:]):
            random = np.random.rand(x, y)
            self.weight.append(random)
        return

    def feed_forward(self, input):
        for b, w in zip(self.biases, self.weight):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def cost_gradient(self, neuronas, expectancy):
        return neuronas - expectancy

    def delta_precision(self, resultado, experanza):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                       for (x, y) in resultado]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivate(x):
    return sigmoid(x) * (1-sigmoid(x))




def back_propagation():
    return





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

data = np.load('../data/all_data.npy')
cant_data = len(data)
batches = []

#PArtir en 3 el dataset y revolverlo
int1 = int(np.round(0.70*cant_data))
int2 = int(np.round(0.85*cant_data))
int3 = cant_data -1

np.random.shuffle(data)

train = data[:int1]
test = data[int1:int2]
CV = data[int2:]

RR([784, 189, 91, 10])




