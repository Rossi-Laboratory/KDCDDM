# models/vae.py
class Encoder(nn.Module):
    """
    VAE Encoder for compressing image into latent representation.
    """
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), Swish(),
            nn.Conv2d(32, 64, 4, 2, 1), Swish(),
            nn.Conv2d(64, 128, 4, 2, 1), Swish()
        )
        self.fc_mu = nn.Linear(128 * 62 * 62, latent_dim)
        self.fc_logvar = nn.Linear(128 * 62 * 62, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    """
    VAE Decoder for reconstructing image from latent vector.
    """
    def __init__(self, latent_dim=128, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 62 * 62)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), Swish(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 62, 62)
        return self.net(h)
