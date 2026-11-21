
import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self):
        super(MLPDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.network(z)
        x = x.view(-1, 1, 28, 28)
        return x
