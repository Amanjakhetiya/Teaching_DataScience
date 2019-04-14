# Ref: Pytorch zero to all lectures by Sung Kim https://www.youtube.com/watch?v=ogZi5oIo4fI
import torch
import torch.nn as nn
from torch.autograd import Variable

# One hot encoding
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

char2id = {'h':0,'e':1,'l':2,'o':3}
input_dim = 4 # Size of the word2vec or one-hot vector
batch_size = 3 # one word at a time
sequence_length = 5 # one character at a time
output_size = 2 # [[[x,x]]] shape (1,1,output_size)

cell = nn.RNN(input_size=input_dim,hidden_size=output_size, batch_first=True)

# One letter input
inputs = Variable(torch.Tensor([[h,e,l,l,o],[e,o,l,l,l],[l,l,e,l,l]])) # shape (1,sequence_length,input_dim)

# Initialize the hidden state # num_layers * num_directions
hidden = Variable(torch.randn(1,batch_size,output_size))

out, hidden = cell(inputs,hidden)
print("Output: ",out)