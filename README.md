Classifying audio tracks based on the Raga(m) (https://en.wikipedia.org/wiki/Raga).

Uses Keras, Theano and Librosa

I have selected two ragams from Carnatic music, Mayamalavagowlai and Shankharabharanam.
Split some tracks from www.sangeethapriya.org into 20 second intervals and extracted chromagrams.
Each track was thus represented as a matrix.

CNN with 2 Conv-Pool layers was applied. Test accuracy of 94.0% was obtained (lots of scope for improvement).
