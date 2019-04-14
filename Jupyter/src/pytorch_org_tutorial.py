# Ref : http://pytorch.org/tutorials/beginner/blitz/

import torch
from torch.autograd import Variable

x = torch.randn(3)
x = Variable(x,requires_grad=True)
print(x.grad)

y = x*2
while y.data.norm() < 1000:
    y = y*2

gradients = torch.FloatTensor([0.1,1,0.0001])
y.backward(gradients)

print(x.grad)