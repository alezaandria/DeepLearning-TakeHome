
import torch
import torch.nn as nn


#Generator =
class Generator(nn.Module):

    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        generated = self.network(z)
        generated = generated.view(-1, 1, 28, 28)
        return generated


#Discriminator =
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(

            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.network(x)
        return output
