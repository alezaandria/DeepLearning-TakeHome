
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):

    def __init__(self):
        super(MLPClassifier, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        h1 = self.relu(self.fc1(x))
        z = self.relu(self.fc2(h1))
        output = self.fc3(z)

        return output, z
