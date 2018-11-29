# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import os
import random

import matplotlib.pyplot as plt
import pandas
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler






class FourierModel(nn.Module):

    def __init__(self, size):
        super(FourierModel, self).__init__()
        self.size = size
        dtype = torch.float
        self.linear1 = torch.randn(2, size, dtype=dtype, requires_grad=True)
        self.linear2 = torch.randn(2, size, dtype=dtype, requires_grad=True)
        self.b1 = torch.randn(size, dtype=dtype, requires_grad=True)
        self.b2 = torch.randn(size, dtype=dtype, requires_grad=True)
        self.c1 = torch.randn(size, 1, dtype=dtype, requires_grad=True)
        self.c2 = torch.randn(size, 1, dtype=dtype, requires_grad=True)
        self.c = torch.randn(1, 1, dtype=dtype, requires_grad=True)
        self.T = torch.randn(1, 1, dtype=dtype, requires_grad=True)

    def forward(self, x):
        x = torch.cat((x % 30, x.mod(self.T)), 1)
        y = torch.cos(x.mm(self.linear1) + self.b1).mm(self.c1) + torch.sin(x.mm(self.linear2) + self.b2).mm(self.c2) + self.c
        return y




dataframe = pandas.read_csv('/home/nazar/PycharmProjects/ptbdb_normal.csv', engine='python').values

row = dataframe[1, 10:150]

T = len(row)

model = FourierModel(30)

x, dx = np.linspace(0, T, T, endpoint=False, retstep=True)

x = torch.tensor(x, dtype = torch.float).reshape(len(x), 1)
y = torch.tensor(row, dtype = torch.float).reshape(len(x), 1)

learning_rate = 0.3
optimizer = torch.optim.Adagrad([model.T, model.c, model.c1, model.c2, model.linear2, model.linear1, model.b1, model.b2], lr=learning_rate)

loss_fn = torch.nn.MSELoss(reduction='sum')

plt.plot(row)
plt.show()

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    with torch.no_grad():
        if t % 100 == 0:
           plt.plot(y_pred.reshape(len(y)).detach().numpy())
           plt.plot(y.numpy())
           plt.show()





