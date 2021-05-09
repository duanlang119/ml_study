#Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow、Theano后端。所以在安装keras的前面，要先搭建tensorflow环境和安装https://www.tensorflow.org/install/install_windows（这里是官方的安装过程）

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 将数据reshape，CNN的输入是4维的张量（可看做多维的向量），第一维是样本规模，第二维是像素通道，第三维和第四维是长度和宽度。并将数值归一化和类别标签向量化。
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28,1), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128, verbose=2)
# Final evaluation of the model

filename='keras_mnistmodel2.h5'
model.save(filename)
scores = model.evaluate(X_test, y_test, verbose=0)
#准确度为
print(scores[1])
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

