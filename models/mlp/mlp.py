import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, input)
        x = self.relu(self.fc1())

