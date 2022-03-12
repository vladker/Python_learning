import  matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import  pandas as pd

from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
print("Keys iris_dataset : \n{}".format(iris_dataset.keys()))
print("Type of array data: {}".format(type(iris_dataset['data'])))
print("Form of array data: {}".format(iris_dataset['data'].shape))
print("Goal : {}".format(iris_dataset['target']))
print("Target names : {}".format(iris_dataset['target_names']))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Features names : \n{}".format(iris_dataset['feature_names']))
print("File location : \n{}".format(iris_dataset['filename']))
print("Top 5 rows of array: \n{}".format(iris_dataset['data'][:5]))
print("Target answers: \n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("Shape of X_train: {}".format(X_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print("Shape of X_test: {}".format(X_test.shape))
print("Shape of y_train: {}".format(y_test.shape))

# Create and training of classificator
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Practical use of classificator
X_new = np.array([[5, 2.9, 1, 0.2]])
pr = knn.predict(X_new)

print("Marker: {}".format(pr))
print("Type: {}".format(iris_dataset['target_names'][pr]))

pr = knn.predict(X_test)
print("Forecast of type in test array: \n{}".format(pr))
print("Precision of forecast on test array: {:.2f}".format(np.mean(pr == y_test)))

# Scatter plot matrix Seaborn
df = sb.load_dataset('iris')
sb.set_style('ticks')
sb.pairplot(df, hue='species', diag_kind='kde', kind='scatter', palette='husl')
plt.show()

# Accuracy of model
pr = knn.predict(X_test)
print('Forecast on test array: \n{}'.format(pr))
print('Accuracy: {:.2f}'.format(np.mean(pr == y_test)))
