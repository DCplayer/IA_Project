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

    def validation(self, set):
        positivos = 0
        for x, y in set:
            z = np.argmax(self.feed_forward(x))
            if y[z] == 1:
                positivos += 1
        return positivos

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
        self.alphas.clear()
        self.deltas.clear()

        self.alphas.append(ingreso)
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

        lista = copy.deepcopy(self.alphas)
        self.alphas.clear()
        for i in range(len(self.deltas) -1):
            self.alphas.append(self.deltas[i] @ np.transpose(lista[i]))

    def descensograd(self, training, test, cross, rate, iter):
        k = rate / len(training)
        gradient_weights = []
        gradient_biases = []

        for i in range(iter):
            contadores = 0

            for x, y in training:
                self.backpropagation(x, y)
                if contadores == 0:
                    gradient_weights = copy.deepcopy(self.alphas)
                    gradient_biases = copy.deepcopy(self.deltas)
                    contadores +=1
                else:
                    for j in range(len(self.alphas)):
                        gradient_weights[j] += self.alphas[j]
                        gradient_biases[j] += self.deltas[j]
            for z in range(len(self.weigths)):
                self.weigths[z] -= k * gradient_weights[z]
                self.biases[z] -= k * gradient_biases[z]
            gradient_weights.clear()
            gradient_biases.clear()

        print("cross_validation: {}%".format(self.success(cross) / len(cross) * 100))
        print("test_validation: {}%".format(self.success(test) / len(test) * 100))
