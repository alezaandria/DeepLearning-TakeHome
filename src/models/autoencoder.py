
import torch
import torch.nn as nn


class VanillaAutoencoder(nn.Module):

    def __init__(self, latent_dim=64):
        super(VanillaAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, latent_dim)
        )

        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):

        #ubah gambar menjadi latent representation
        z = self.encoder(x)

        #mengembalikan latent representation menjadi gambar
        reconstructed = self.decoder(z)

        #ngubah bentuk menjadi [batch, 1, 28, 28]
        reconstructed = reconstructed.view(-1, 1, 28, 28)

        return reconstructed, z
