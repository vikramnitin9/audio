# 1379 samples

import numpy as np

from six.moves import cPickle

import librosa
import librosa.display

data = np.zeros([39*80, 1292, 12])

count = 0

for i in range(1,40):
	audio_path = 'data/M/'+str(i)+'.mp3'

	y, sr = librosa.load(audio_path)

	print(i, " loaded")

	dur = np.shape(y)[0]
	samp = 30*sr

	print("Duration of track : ", (int)(dur/sr))
	print("Number of samples : ", (int)(dur/samp))

	for j in range(0,(int)(dur/samp)):
		y_curr = y[j*samp : (j+1)*samp]
		S = librosa.feature.melspectrogram(y_curr, sr=sr, n_mels=128)
		log_S = librosa.logamplitude(S, ref_power=np.max)
		mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=12)
		data[count] = np.transpose(mfcc)
		count += 1

data = data[:count]

print("Total number of samples : ", count)

file =  open("data/M.pkl", 'wb')
cPickle.dump(data,file,cPickle.HIGHEST_PROTOCOL)
file.close()
