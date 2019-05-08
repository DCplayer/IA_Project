import matplotlib.pyplot as plt
import numpy as np


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
print(len(data))



peso1 = np.random.randn()
peso2 = np.random.randn()
bias = np.random.randn()


def sigmoid(x):
    return 1/(1 + np.exp(-x))
