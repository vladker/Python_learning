from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import logging
import numpy as np

model = Sequential()
model.add(Dense(6, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1))
print(model.summary())

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
model.fit(X, y, batch_size=1, epochs=1000, verbose=0)

print("Проверка работы обученной сети:")
print("XOR (0,0):", model.predict(np.array([[0, 0]])))
print("XOR (0,1):", model.predict(np.array([[0, 1]])))
print("XOR (1,0):", model.predict(np.array([[1, 0]])))
print("XOR (1,1):", model.predict(np.array([[1, 1]])))

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