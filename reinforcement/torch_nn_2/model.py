import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Deep_Q_Network(nn.Module):
    def __init__(self, lr, input_dims, f1_dims, f2_dims, n_actions):
        super(Deep_Q_Network, self).__init__()
        self.input_dims = input_dims
        self.f1_dims = f1_dims
        self.f2_dims = f2_dims
        self.n_actions = n_actions
        self.f1 = nn.Linear(*self.input_dims, self.f1_dims)
        self.f2 = nn.Linear(self.f1_dims, self.f2_dims)
        self.f3 = nn.Linear(self.f2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))
        actions = self.f3(x)

        return actions