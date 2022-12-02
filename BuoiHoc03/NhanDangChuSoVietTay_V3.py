import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2

model_architecture = "digit_config.json"
model_weights = "digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 

optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 

image = np.zeros((10*28, 15*28), dtype = np.uint8)

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
index = np.random.randint(0, 9999, 150)
for i in range(0, 150):
    m = i // 15
    n = i % 15
    image[m*28:(m+1)*28, n*28:(n+1)*28] = X_test[index[i]] 

#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')

#normalize in [0,1]
X_test /= 255

X_test_image = np.zeros((150,784), dtype = np.float32)
for i in range(0, 150):
    X_test_image[i] = X_test[index[i]] 

prediction = model.predict(X_test_image)
for i in range(0, 150):
    ket_qua = np.argmax(prediction[i])
    print(ket_qua, end = ' ')
    if (i+1) % 15 == 0:
        print()

cv2.imshow('Image', image)
cv2.waitKey(0)

