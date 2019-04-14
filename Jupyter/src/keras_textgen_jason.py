# Ref: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename ="data/wonderland.txt"
modelpath = "models/wts.hdf5"


raw_text = open(filename,"r", encoding="utf-8").read()
raw_text = raw_text.lower()

# Create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c,i) for i , c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
# print(n_chars)
# print(n_vocab)

'''
Each training pattern of the network is comprised of 100 time steps of 
one character (X) followed by one character output (y). 
When creating these sequences, we slide this window along the whole book 
one character at a time, allowing each character a chance to be learned 
from the 100 characters that preceded it (except the first 100 characters of 
course).
------<100 chars as x>------- => 101th char as y,and this window slides
'''

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([chars_to_int[cc] for cc in seq_in])
    dataY.append(chars_to_int[seq_out])
n_patterns = len(dataX)
# print(n_patterns)

# dataX = numpy.array(dataX)

def train():
    # Transform input to make it suitable for Keras  [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns,seq_length,1))
    # Normalize
    X = X/float(n_vocab) # makes it 0 to 1, with floats in it
    # One hot encoding of the target
    y = np_utils.to_categorical(dataY)

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')


    # define the checkpoint
    filepath = "models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X,y,epochs=5, batch_size=128, callbacks=callbacks_list)

    model.save(modelpath)
    return model

def test(model):
    int_to_chars = dict((i,c) for i, c in enumerate(chars))

    start = numpy.random.randint(0,len(dataX)-1)
    pattern = dataX[start]
    print("Seed : {}".format("".join([int_to_chars[ii] for ii in pattern])))
    # generate characters
    for i in range(100):
        x = numpy.reshape(pattern, (1,len(pattern),1))
        x = x/float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_chars[index]
        print(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

from keras.models import load_model
import os

if __name__ == "__main__":
    model = None
    if not os._exists(modelpath):
        model = train()
    else:
        model = load_model(modelpath)

    if not model == None:
        test(model)

