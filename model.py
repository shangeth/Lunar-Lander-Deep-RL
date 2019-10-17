import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        middle_size1 = 300
        middle_size2 = 300
        self.fc1 = nn.Linear(state_size, middle_size1)
        self.fc2 = nn.Linear(middle_size1, middle_size2)
        self.fc3 = nn.Linear(middle_size2, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
