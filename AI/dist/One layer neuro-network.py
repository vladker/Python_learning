import  numpy as np
#Activator
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

#Neuron description
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# 3 layers network description
class OurNeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1]) #all weights are the same for all
        bias = 0 #bias is the same for all

        #create a network
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x) #формируем выхода Y1 из нейрона h1
        out_h2 = self.h2.feedforward(x) #формируем выхода Y2 из нейрона h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2])) #формируем выход Y из нейрона o1
        return out_o1
network= OurNeuralNetwork() #создаем сеть из класса
x = np.array([2,3]) #формируем входные параметры сети X1=2, X2=3
print('Y= ', network.feedforward(x)) #передаем входы в сеть