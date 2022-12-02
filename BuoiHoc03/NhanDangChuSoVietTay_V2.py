import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2


mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

index = 1000
sample = X_test[index]

#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize in [0,1]
X_train /= 255
X_test /= 255

#load model
model_architecture = "digit_config.json"
model_weights = "digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 
#train
optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 

X_test_sample = X_test[index] 
X_test_sample = X_test_sample.reshape(1, RESHAPED)
#predict
prediction = model.predict(X_test_sample)
ket_qua = np.argmax(prediction)
print(ket_qua)

cv2.imshow('Digit', sample)
cv2.waitKey(0)

