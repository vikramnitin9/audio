# 1356 samples

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
from six.moves import cPickle

ms.use('seaborn-muted')

import librosa
import librosa.display

data = np.zeros([31*80, 352, 12])

count = 0

for i in range(1,32):
	audio_path = 'data/M/'+str(i)+'.mp3'

	y, sr = librosa.load(audio_path, sr=9000)

	print("Samping rate : ", sr)
	print(i, " loaded")

	dur = np.shape(y)[0]
	samp = 20*sr

	print("Duration of track : ", (int)(dur/sr))
	print("Number of samples : ", (int)(dur/samp))

	for j in range(0,(int)(dur/samp)):
		y_curr = y[j*samp : (j+1)*samp]
		C = librosa.feature.chroma_cqt(y=y_curr, sr=sr)
		data[count] = np.transpose(C)
		count += 1

data = data[:count]

print("Total number of samples : ", count)

file =  open("data/M.pkl", 'wb')
cPickle.dump(data,file,cPickle.HIGHEST_PROTOCOL)
file.close()
