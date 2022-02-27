import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Dataset')
print(df.to_string())

# Select DF 100 objects
y = df.iloc[0:100, 4].values
print('4th column values')
print(y)

# Flowers to binary names
y = np.where(y == "Iris-setosa", -1, 1)
print('Flowers names values')
print(y)

# Select 100 objects from dataset into matrix
X = df.iloc[0:100, [0, 2]].values
print('Values X - 100')
print(X)
print('End of X')

# Values to plot
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Multicolor')

# Axis names
plt.xlabel('Lenght cup')
plt.ylabel('Lenght wing')
plt.legend(loc='upper left')
plt.show()


# Description of perceptron
class Perceptron(object):
    '''
    eta:float - speed
    n_iter:int - rounds
    w_:1-axis array, weight after adjustment
    errors_: list of errors of classification
    '''

    def __init__(self, eta=0.01, n_iter=10):
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
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
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
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epoches')

'''Errors of classification'''
plt.ylabel('Errors of classification')
plt.show()

'''Visualisation of border'''
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup of markers and palette
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # show plot of decisions
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #     show class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=cl)


# Draw picture
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Length cup, sm')
plt.ylabel('Length wing, sm')
plt.legend(loc='upper left')
plt.show()

# Adaptive linear neuron
class AdaptiveLinearNeuron(object):
    '''
    Classificator based on ADALINE
    Params
    eta:float - speed of learning
    n_iter:int round
    Attributes
    w_:1 axe array - weights after training
    errors_: list -number of errors of classification in each epoch
    '''
    def __init__(self, rate=0.01, n_iter=10):
        self.rate=rate
        self.n_iter=n_iter

    def fit(self, X, y):
        '''

        :param X: array, structure =[n_samples, n_features] - training vectors
        :param y: array, structure =[n_samples] - target values
        :return: self object
        '''
        self.weight = np.zeros(1+X.shape[1])
        self.cost = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.weigth[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self
