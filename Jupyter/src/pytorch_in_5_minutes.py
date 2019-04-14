# Ref: https://www.youtube.com/watch?v=nbJ-2G2GXL0&vl=en
# https://github.com/llSourcell/pytorch_in_5_minutes/blob/master/demo.py
# Imperative Programming: Tell what and tell how
# Symbolic: define computation graph first, then supply actual values to compute the result.
# Dynamic Computation Graph: Graph gets defined at the run time.
# Static graphs work well for fixed size networks, NN or CNN but not RNN or LSTM


import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

# N is batch size
# D_in is input dimension
# H is hidden dimension
# D_out is output dimension

N, D_in,H,D_out = 64,1000,100,10

# Create random tensors to hold inputs and outputs
# requires_grad=False indicates that we do not need to compute gradients for these

x = Variable(torch.randn(N,D_in).type(dtype), requires_grad = False)
y = Variable(torch.randn(N,D_out).type(dtype), requires_grad = False)
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

print(x)
print(y)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(t,loss.data[0])
    loss.backward() #  w1.grad and w2.grad will be Variables holding the gradient

    # Update weights using gradient descent; w1.data and w2.data are Tensors,
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
    # Tensors.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # manually zero the gradients
    w1.grad.data.zero_()
    w2.grad.data.zero_()