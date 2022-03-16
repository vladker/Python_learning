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
from itertools import permutations

def model_summary():
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

# load model
model = tf.keras.models.load_model('C:/Users/rogac/PycharmProjects/Git/AI/Tensorflow+keras')


test_list = list()
for sequence in permutations('010', 3):
    fixed_sequence = []
    for elements in sequence:
        elements = int(elements)
        fixed_sequence.append(elements)
    test_list.append(fixed_sequence)
for sequence in permutations('101', 3):
    fixed_sequence = []
    for elements in sequence:
        elements = int(elements)
        fixed_sequence.append(elements)
    test_list.append(fixed_sequence)
for tests in test_list:
    print(tests, model.predict(np.array([tests])))



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
