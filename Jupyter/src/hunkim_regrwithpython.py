import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

def gradient(x,y):
    return 2*x*(x*w-y)

w = 1.0

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val,y_val)
        w = w - 0.01 * grad
        l = loss(x_val,y_val)
    print("Epoch=", epoch, " Loss= ",l)

print("Result for x = 4: ", forward(4))