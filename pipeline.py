from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.optimizers import SGD

from six.moves import cPickle

import numpy as np
import librosa


print("Loading audio track...")
audio_path = 'data/input.mp3'
y, sr = librosa.load(audio_path, sr=9000)
print("Loaded!")
print()
print("Samping rate : ", sr)

dur = np.shape(y)[0]
samp = 20*sr

num_test_samples = (int)(dur/samp)

print("Duration of track : ", (int)(dur/sr))
print("Number of 20s samples : ", num_test_samples)

max_len = 352 # 352, 9000, 20s seems to be a valid combo. No idea why
no_of_features = 12

X_test = np.zeros([num_test_samples,max_len,no_of_features])

print("Constructing chromagrams...")

for j in range(0,num_test_samples):
	y_curr = y[j*samp : (j+1)*samp]
	C = librosa.feature.chroma_cqt(y=y_curr, sr=sr)
	X_test[j] = np.transpose(C)

print("Loading pixel-wise means and subtracting...")

file = open("means.pkl","rb")
pixel_mean = cPickle.load(file)
file.close()

for i in range(0,num_test_samples):
    X_test[i] -= pixel_mean

X_test = X_test.reshape((num_test_samples,1,max_len,no_of_features))

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

print("Model compiled!")
print("Loading saved weights...")
model.load_weights('MS.h5')

print("Predicting...")

y_pred = model.predict(X_test)

print(y_pred)

val = np.mean(y_pred)

print()

if(val > 0.5):
	print("Shankharabharanam. Confidence : ", val*100,"%")
else:
	print("Mayamalavagowlai. Confidence : ", (1-val)*100,"%")
