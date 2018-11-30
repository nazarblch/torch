# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

        self.linear1 = torch.randn(1, size, dtype=dtype, requires_grad=True)
        self.linear2 = torch.randn(1, size, dtype=dtype, requires_grad=True)
        self.b1 = torch.randn(size, dtype=dtype, requires_grad=True)
        self.b2 = torch.randn(size, dtype=dtype, requires_grad=True)
        self.c1 = torch.randn(size, 1, dtype=dtype, requires_grad=True)
        self.c2 = torch.randn(size, 1, dtype=dtype, requires_grad=True)
        self.c = torch.randn(1, 1, dtype=dtype, requires_grad=True)

        self.linear3 = torch.randn(1, 10, dtype=dtype, requires_grad=True)
        self.b3 = torch.randn(10, dtype=dtype, requires_grad=True)
        self.c3 = torch.randn(10, 1, dtype=dtype, requires_grad=True)

        self.linear4 = torch.ones(1, 10, dtype=dtype, requires_grad=True)
        self.linear5 = torch.ones(1, 10, dtype=dtype, requires_grad=True)
        self.b4 = torch.randn(10, dtype=dtype, requires_grad=True)
        self.c4 = torch.randn(10, 1, dtype=dtype, requires_grad=True)



        self.optimizer1 = torch.optim.Adagrad(
            [self.linear2, self.linear1, self.linear3, self.linear4, self.linear5, self.b1, self.b2, self.b3, self.b4],
            lr=0.2)
        self.optimizer2 = torch.optim.Adagrad(
            [self.c, self.c1, self.c2, self.c3, self.c4],
            lr=0.4)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, x):
        y = torch.cos(x.mm(self.linear1) + self.b1).mm(self.c1) + \
            torch.relu(torch.cos(x.mm(self.linear3) + self.b3)).mm(self.c3) + \
            torch.sin(x.mm(self.linear2) + self.b2).mm(self.c2) + \
            self.c

        return y


    def train(self, data, steps):

        loss = 0
        y_pred = None

        n = len(data)

        x, dx = np.linspace(0, n, n, endpoint=False, retstep=True)

        x = torch.tensor(x, dtype=torch.float).reshape(len(x), 1)
        y = torch.tensor(data, dtype=torch.float).reshape(len(x), 1)

        all_linear1_params = torch.cat([p.view(-1) for p in [self.c1, self.c2]])

        for t in range(steps):

            y_pred = self.forward(x)
            l1_regularization = 0.01 * torch.norm(all_linear1_params, 1)
            loss = self.loss_fn(y_pred, y)
            L1loss = loss + l1_regularization

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            L1loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

        return loss, y_pred






dataframe = pandas.read_csv('data/ptbdb_normal.csv', engine='python').values

dataframe1 = pandas.read_csv('data/ptbdb_abnormal.csv', engine='python').values

row = dataframe[1, 10:150]
row1 = dataframe1[10, 10:150]
model = FourierModel(100)


plt.plot(np.concatenate((row, row)))
plt.show()

loss, y_pred = model.train(np.concatenate([row, row]), 1000)

print(loss)


plt.plot(y_pred.detach().numpy())
plt.show()