import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header = None)
print('Dataset')
print(df.to_string())

# Select DF 100 objects
y = df.iloc[0:100, 4].values
print('4th column values')
print(y)

# Flowers to binary names
y = np.where(y == "Iris-setosa", -1,1)
print('Flowers names values')
print(y)

# Select 100 objects from dataset into matrix
X = df.iloc[0:100, [0,2]].values
print('Values X - 100')
print(X)
print('End of X')

# Values to plot
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker = 'o', label = 'Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'Multicolor')

# Axis names
plt.xlabel('Lenght cup')
plt.ylabel('Lenght wing')
plt.legend(loc = 'upper left')
plt.show()

# Description of perceptron
class Perceptron(object):
    '''
    eta:float - speed
    n_iter:int - rounds
    w_:1-axis array, weight after adjustment
    errors_: list of errors of classification
    '''

    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    '''
    Adjust model to learning array
    X: array with format = [n_samples, n_features] training vectors
    n_samples - examples
    n_features - features
    
    y: array with format = [n_samples] target values
    returns 
    self: object
    '''

    def fit(self, x, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    '''
    Evaluate clean input
    '''
    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    '''Return class marker after single jump'''
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

'''Training'''
ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epoches')

'''Errors of classification'''
plt.ylabel('Errors of classification')
plt.show()

