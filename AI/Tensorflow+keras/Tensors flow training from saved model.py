from numpy import genfromtxt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import logging
import numpy as np

# Read to np array
file = genfromtxt('H1RDBD.csv', delimiter=',')
X = []
y = []
for rows in file:
    counter = 0
    input_list_X = []
    input_list_y = []
    for element in rows:
        if counter <= 2:
            input_list_X.append(element)
            counter += 1
        else:
            input_list_y.append(element)
            break
    X.append(input_list_X)
    y.append(input_list_y)

model = tf.keras.models.load_model('C:/Users/rogac/PycharmProjects/Git/AI/Tensorflow+keras')
# Параметры 1 уровня
W1 = model.get_weights()[0]
b1 = model.get_weights()[1]

# Параметры 2 уровня
W2 = model.get_weights()[2]
b2 = model.get_weights()[3]

print('W1:', W1)
print('b1:', b1)
print('W2:', W2)
print('b2:', b2)

model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.0001))

# Параметры 1 уровня
W1 = model.get_weights()[0]
b1 = model.get_weights()[1]

# Параметры 2 уровня
W2 = model.get_weights()[2]
b2 = model.get_weights()[3]

print('W1:', W1)
print('b1:', b1)
print('W2:', W2)
print('b2:', b2)

print(model.summary())
model.fit(X, y, batch_size=1, epochs=100, verbose=1)

model.save('C:/Users/rogac/PycharmProjects/Git/AI/Tensorflow+keras')
print("[1, 1, 1]", model.predict(np.array([[1, 1, 1]])))
print("[0, 0, 0]", model.predict(np.array([[0, 0, 0]])))
print("[1, 0, 0]", model.predict(np.array([[1, 0, 0]])))
print("[0, 1, 0]", model.predict(np.array([[0, 1, 0]])))
print("[0, 0, 1]", model.predict(np.array([[0, 0, 1]])))
print("[1, 0, 1]", model.predict(np.array([[1, 0, 1]])))
print("[1, 1, 0]", model.predict(np.array([[1, 1, 0]])))
print("[0, 1, 1]", model.predict(np.array([[0, 1, 1]])))


# Параметры 1 уровня
W1 = model.get_weights()[0]
b1 = model.get_weights()[1]

# Параметры 2 уровня
W2 = model.get_weights()[2]
b2 = model.get_weights()[3]

print('W1:', W1)
print('b1:', b1)
print('W2:', W2)
print('b2:', b2)
