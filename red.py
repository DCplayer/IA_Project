import matplotlib.pyplot as plt
import numpy as np
import copy

class RR:
    def __init__(self, tamanio):
        self.weigths = []
        self.biases = []
        self.alphas = []
        self.deltas = []

        if len(tamanio) > 1:
            for i in range(len(tamanio) - 1):
                w1 = np.random.rand(tamanio[i + 1], tamanio[i])
                b1 = np.random.randn(tamanio[i + 1], 1)

                self.weigths.append(w1)
                self.biases.append(b1)
        else:
            print("La red neuronal debe de tener al menos una capa, aparte de la inicial")
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, x):
        for i in range(len(self.weigths)):
            a = self.sigmoid((self.weigths[i] @ x) + self.biases[i])
            x = a
        return x

    def backpropagation(self, ingreso, prediction):
        counter = 0
        d = 0
        for i in range(len(self.weigths)):
            a = self.sigmoid((self.weigths[i] @ ingreso) + self.biases[i])
            self.alphas.append(a)
            ingreso = copy.deepcopy(a)

            if counter == 0:
                counter = counter + 1
                d = self.alphas[len(self.alphas) - 1] - prediction
                self.deltas.append(d)
            else:
                d = np.transpose(self.weigths[i]) @ self.deltas[i - 1] * self.alphas[i] * (1 - self.alphas[i])
                self.deltas.append(d)
        self.deltas.reverse()

    def descenso_grad(self):
        pass

