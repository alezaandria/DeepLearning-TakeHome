
import torch
import torch.nn as nn


#Generator
class DCGANGenerator(nn.Module):

    def __init__(self, noise_dim=100):
        super(DCGANGenerator, self).__init__()

        self.network = nn.Sequential(

            # 1 x 1 -> 7 x 7
            nn.ConvTranspose2d(
                noise_dim, 128,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 7 x 7 -> 14 x 14
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 14 x 14 -> 28 x 28
            nn.ConvTranspose2d(
                64, 1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)


# Discriminator
class DCGANDiscriminator(nn.Module):

    def __init__(self):
        super(DCGANDiscriminator, self).__init__()

        self.network = nn.Sequential(

            # 28 x 28 -> 14 x 14
            nn.Conv2d(
                1, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 14 x 14 -> 7 x 7
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 7 x 7 -> 1 x 1
            nn.Conv2d(
                128, 1,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)
