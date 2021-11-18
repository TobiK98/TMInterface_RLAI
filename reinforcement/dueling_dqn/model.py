import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import os


class DuelingLinearDeepQNetwork(nn.Module):
    def __init__(self, alpha, n_actions, name, input_dims, chkpt_dir='tmp/dqn'):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'dueling_dqn')

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A

    def save_checkpoint(self):
        print('..... saving checkpoint .....')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('..... loading checkpoint .....')
        self.load_state_dict(T.load(self.checkpoint_file))