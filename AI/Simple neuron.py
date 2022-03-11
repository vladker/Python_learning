# Simple Neuron
import numpy as np

# activation function: f(x) = 1 / (1 + e**(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Creation of class neuron

class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def y(self, x): #сумматор
        s = np.dot(self.w, x) + self.b #суммирует входы
        return sigmoid(s) #обращение к функции активации

Xi = np.array([2, 3]) #Задание значений входам x1 = 2, x2 = 3
Wi = np.array([0, 1]) #Веса входных сенсоров w1 = 0, w2 = 1
bias = 4 #смещение
n = Neuron(Wi, bias) #создание объекта класса Нейрон
print('Y=', n.y(Xi)) #обращение к нейрону
