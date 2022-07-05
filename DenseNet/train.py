import tensorflow as tf
import numpy as np
from modules.MnistData import MnistData
from modules.Build_model import Build_net


data = MnistData('./mnist.npz')

x_train, y_train = data.x_train, data.y_train
x_test, y_test = data.x_test, data.y_test

x_train = x_train.reshape(
    [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
x_test = x_test.reshape([x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])

model = Build_net([28, 28], 3, 32)
model.summary()

epoch = 10
batch_size = 100

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,  verbose=1,
          validation_split=0.1)

model.evaluate(x_test,  y_test, verbose=2)
