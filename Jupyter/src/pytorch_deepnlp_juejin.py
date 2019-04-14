import torch
#import torch.autograd as autograd
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

torch.manual_seed(1)

V_data = [1.,2.,3.]
V = torch.Tensor(V_data)
print(V)

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.Tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2.
T_data = [[[1.,2.], [3.,4.]],
          [[5.,6.], [7.,8.]]]
T = torch.Tensor(T_data)
print(T)

print(V[0])
print(M[0])
print(T[0])

x = torch.randn((3, 4, 5))
print(x)

x = torch.Tensor([1.,2.,3.])
y = torch.Tensor([4.,5.,6.])
z = x + y
print(z)

# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 =torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1) # second arg specifies which axis to concat along
print(z_2)

# If your tensors are not compatible, torch will complain.  Uncomment to see the error
# torch.cat([x_1, x_2])

x = torch.randn(2,3,4)
print(x)
print(x.view(2,12)) # Reshape to 2 rows, 12 columns
print(x.view(2, -1)) # Same as above.  If one of the dimensions is -1, its size can be inferred
#
#x = autograd.Variable(torch.Tensor([1.,2.,3.]), requires_grad = True)
#print(x.data)
#
#y = autograd.Variable( torch.Tensor([4., 5., 6]), requires_grad=True )
#z = x + y
#print(z.data)
#
#print(z.grad_fn)
## Lets sum up all the entries in z
#s = z.sum()
#print(s)
#print(s.grad_fn)
#
#s.backward() # calling .backward() on any variable will run backprop, starting from it.
#print(x.grad)

x = torch.Tensor([1.,2.,3.])
y = torch.Tensor([4.,5.,6.])
x.requires_grad = True
y.requires_grad = True
z = x + y
print(z.data)
print(z.grad_fn)
s = z.sum()
print(s.data)
print(s.grad_fn)
s.backward() # calling .backward() on any variable will run backprop, starting from it.
print(x.grad)

x = torch.randn((2,2))
y = torch.randn((2,2))
x.requires_grad = True
y.requires_grad = True
z = x + y 
print(z.grad_fn)

var_z_data = z.data # Get the wrapped Tensor object out of var_z...
new_var_z = torch.Tensor( var_z_data ) # Re-wrap the tensor in a new variable

# does new_var_z have information to backprop to x and y? NO!
print(new_var_z.grad_fn)
# And how could it?  We copied the tensor values out of z (that is what z.data is).  This tensor doesn't know anything about how it was computed.   If var_z_data doesn't know how it was computed, theres no way new_var_z will.
data = torch.randn(5,requires_grad=True)
print(data)
import torch.nn.functional as F
print(F.relu(data))
print(F.softmax(data,0))
print(F.softmax(data,0).sum())
print(F.log_softmax(data,0))
