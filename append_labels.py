import numpy as np
from six.moves import cPickle

print("M loaded")

file = open("data/M.pkl","rb")
M_X = cPickle.load(file)
file.close()

print("S loaded")

file = open("data/S.pkl","rb")
S_X = cPickle.load(file)
file.close()

M_y = np.zeros([np.shape(M_X)[0],1])
S_y = np.ones([np.shape(S_X)[0],1])

X = np.vstack([M_X, S_X])
y = np.vstack([M_y, S_y])

print("Stacked")

assert len(X) == len(y)

p = np.random.permutation(len(X))
X = X[p]
y = y[p]

print("Shuffled")
print("Writing to file")
file = open("data/chroma.pkl","wb")
cPickle.dump((X,y),file,cPickle.HIGHEST_PROTOCOL)
file.close()
