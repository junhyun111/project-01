
import math

import torch
from torch import nn


class BlendGroupNorm(nn.Module):
    """Learned mixture of identity and 8-group GroupNorm."""

    def __init__(self, channels, initial_fraction):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.gate_logit = nn.Parameter(
            torch.full((1, channels, 1, 1), initial_logit)
        )

    def forward(self, x):
        alpha = torch.sigmoid(self.gate_logit)
        return x + alpha * (self.norm(x) - x)


class ResidualBlock(nn.Module):
    """Same residual block used in the current repository."""

    def __init__(self, channels, normalize="group", initial_fraction=0.5):
        super().__init__()
        if normalize == "group":
            def make_norm():
                return nn.GroupNorm(8, channels)
        elif normalize == "blend":
            def make_norm():
                return BlendGroupNorm(channels, initial_fraction)
        else:
            raise ValueError("normalize must be 'group' or 'blend'")

        self.layers = nn.Sequential(
            make_norm(),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            make_norm(),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.layers(x)


class LatentFiLM(nn.Module):
    """Condition a decoder feature map on the sampled latent z."""

    def __init__(self, channels, latent_dim):
        super().__init__()
        self.affine = nn.Linear(latent_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h, z):
        gain, shift = self.affine(z).chunk(2, dim=1)
        return h * (1 + gain[:, :, None, None]) + shift[:, :, None, None]


class VAE(nn.Module):
    """32D diagonal-Gaussian VAE; exactly film_weak_all_z32 in Kaggle."""

    def __init__(self):
        super().__init__()
        self.latent_dim = 32
        self.widths = (32, 64, 128)
        c1, c2, c3 = self.widths

        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            ResidualBlock(c1),

            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            ResidualBlock(c2),

            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.SiLU(),
            ResidualBlock(c3),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(c3 * 4 * 4, self.latent_dim)
        self.fc_logvar = nn.Linear(c3 * 4 * 4, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, c3 * 4 * 4)

        # Decoder: BlendGroupNorm initial fractions (0.30, 0.30, 0.10).
        self.dec0 = nn.Sequential(
            ResidualBlock(c3, normalize="blend", initial_fraction=0.30)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c3, c2, 3, padding=1, bias=False),
            BlendGroupNorm(c2, initial_fraction=0.30),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            ResidualBlock(c2, normalize="blend", initial_fraction=0.30)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c1, 3, padding=1, bias=False),
            BlendGroupNorm(c1, initial_fraction=0.10),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            ResidualBlock(c1, normalize="blend", initial_fraction=0.10)
        )
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Latent-conditioned FiLM, initialized as the identity.
        self.cond0 = LatentFiLM(c3, self.latent_dim)
        self.cond1 = LatentFiLM(c2, self.latent_dim)
        self.cond2 = LatentFiLM(c1, self.latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = self.fc_decode(z).reshape(z.shape[0], self.widths[2], 4, 4)
        h = self.dec0(h)
        h = self.cond0(h, z)
        h = self.up1(h)
        h = self.dec1(h)
        h = self.cond1(h, z)
        h = self.up2(h)
        h = self.dec2(h)
        h = self.cond2(h, z)
        return self.head(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
