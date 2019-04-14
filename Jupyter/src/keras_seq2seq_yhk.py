# Ref: https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

from random import randint
import numpy as np
import data
import data_utils

DATASETPATH='D:/Yogesh/ToDos/Projects/Research/NIPS2017/CoVe/practical_seq2seq-master/datasets/twitter/'
# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH=DATASETPATH)
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

n_unique = xvocab_size

# def generate_sequence(length, n_unique):
#     return [randint(0, n_unique-1) for _ in range(length)]
#
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq] # gives index of the max value position

# # sequence = generate_sequence(5,50)
# # print(sequence)
# # encoded = one_hot_encode(sequence, 50)
# # print(encoded)
# # decoded = one_hot_decode(encoded)
# # print(decoded)
#
# def get_pair(n_in,n_out, n_unique):
#     sequence_in = generate_sequence(n_in, n_unique)
#     sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
#     # print("Pair {} : {}".format(sequence_in,sequence_out))
#
#     X = one_hot_encode(sequence_in,n_unique)
#     y = one_hot_encode(sequence_out, n_unique)
#
#     X = X.reshape((1,X.shape[0],X.shape[1]))
#     y = y.reshape((1, y.shape[0], y.shape[1]))
#
#     return X, y
#
# X, y = get_pair(5, 2, 50)

# # configure problem
n_features = n_unique
n_timesteps_in = xseq_len
n_timesteps_out = yseq_len
X = np.array([one_hot_encode(xx,n_unique) for xx in trainX])
y = np.array([one_hot_encode(yy,n_unique) for yy in trainY])
X = X.reshape((1,X.shape[0],X.shape[1]))
y = y.reshape((1, y.shape[0], y.shape[1]))
print(X.shape, y.shape)
print('X={}, y={}'.format(X[0],y[0]))
#

# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import TimeDistributed
# from keras.layers import RepeatVector
#
# # model = Sequential()
# # model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
# # model.add(RepeatVector(n_timesteps_in))
# # model.add(LSTM(150, return_sequences=True))
# # model.add(TimeDistributed(Dense(n_features, activation="softmax")))
# #
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# #
#
# from attention_decoder import AttentionDecoder
#
# model = Sequential()
# model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
# model.add(AttentionDecoder(150, n_features))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
#
# for epoch in range(5000):
#     X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#     # print("X {}, y{}".format(X,y))
#     model.fit(X,y, epochs=1, verbose=2)
#
# # evaluate LSTM
# total, correct = 100, 0
# for _ in range(total):
#     X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#     yhat = model.predict(X, verbose=0)
#     if np.array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
#         correct += 1
# print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
#
# # spot check some examples
# for _ in range(10):
#     X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#     yhat = model.predict(X, verbose=0)
#     print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
#
#
