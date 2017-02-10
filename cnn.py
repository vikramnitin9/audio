from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.optimizers import SGD
from keras.regularizers import l2

from keras import backend as K

import theano

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

from six.moves import cPickle

import numpy as np
import matplotlib.pyplot as plt

max_len = 352
no_of_features = 12

model = Sequential()
model.add(Convolution2D(32, 6, 2, border_mode="same", input_shape=(1,max_len,no_of_features),init="lecun_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,1)))

model.add(Convolution2D(64, 6, 2, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 1)))

model.add(Flatten())
model.add(Dense(512, init="lecun_uniform"))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, init="lecun_uniform"))
model.add(Activation("sigmoid"))

sgd = SGD(lr=0.1, momentum=0, nesterov=False)
model.compile(loss="mean_squared_error", optimizer=sgd, metrics=['accuracy'])

file = open("data/chroma.pkl","rb")
(X,y) = cPickle.load(file)

print("Loaded dataset");

(X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.1)

num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]

pixel_mean = np.zeros([max_len,no_of_features])

for i in range(0,num_train_samples):
    pixel_mean += (X_train[i]/num_train_samples)

for i in range(0,num_train_samples):
    X_train[i] -= pixel_mean

for i in range(0,num_test_samples):
    X_test[i] -= pixel_mean

# print(X_train[0:5,0:5,:])

X_train = X_train.reshape((num_train_samples,1,max_len,no_of_features))
X_test = X_test.reshape((num_test_samples,1,max_len,no_of_features))

print("Training set dimensions : ", np.shape(X_train))
print("Training labels dimensions : ", np.shape(y_train))

model.fit(X_train, y_train, batch_size=10, nb_epoch=20, validation_split=0.1)

model.save_weights('MS.h5')

score = model.evaluate(X_test, y_test, verbose=1)

# y_pred = model.predict(X_train)

# for i in range(0,20):
# 	print("Actual : ",y_train[i],"Predicted : ",y_pred[i])

print("Test accuracy : ", score[1]);
