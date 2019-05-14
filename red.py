import matplotlib.pyplot as plt
import numpy as np

class RR:
    def __init__(self, tamanio):
        self.weigths = []
        self.biases = []


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
