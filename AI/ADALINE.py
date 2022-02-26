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