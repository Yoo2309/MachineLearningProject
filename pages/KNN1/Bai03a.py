import tensorflow as tf
from tensorflow import keras 
import numpy as np
import cv2
import joblib

mnist = keras.datasets.mnist 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 


index = np.random.randint(0, 9999, 100)
sample = np.zeros((100,28,28), np.uint8)
for i in range(0, 100):
    sample[i] = X_test[index[i]]


# 784 = 28x28
RESHAPED = 784
sample = sample.reshape(100, RESHAPED) 
knn = joblib.load("knn_mnist.pkl")
predicted = knn.predict(sample)
k = 0
for x in range(0, 10):
    for y in range(0, 10):
        print('%2d' % (predicted[k]), end='')
        k = k + 1
    print()

digit = np.zeros((10*28,10*28), np.uint8)
k = 0
for x in range(0, 10):
    for y in range(0, 10):
        digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
        k = k + 1

cv2.imshow('Digit', digit)
cv2.waitKey(0)



