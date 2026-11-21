
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()), dim=1
        )
        return embeddings


class DDPMModel(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, 128),
            nn.ReLU()
        )

        self.time_layer1 = nn.Linear(128, 64)
        self.time_layer2 = nn.Linear(128, 128)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, time):
        t = self.time_embedding(time)

        t1 = self.time_layer1(t)[:, :, None, None]
        t2 = self.time_layer2(t)[:, :, None, None]

        x = self.conv1(x)
        x = self.relu(x + t1)

        x = self.conv2(x)
        x = self.relu(x + t2)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        return self.output(x)
