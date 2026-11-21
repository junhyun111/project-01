import math
import torch
import torch.nn.functional as F
from torch import nn


class BlendGroupNorm(nn.Module):
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
    def __init__(self, channels, normalize="group", initial_fraction=0.5):
        super().__init__()
        if normalize == "group":
            self.norm1 = nn.GroupNorm(8, channels)
            self.norm2 = nn.GroupNorm(8, channels)
        elif normalize == "blend":
            self.norm1 = BlendGroupNorm(channels, initial_fraction)
            self.norm2 = BlendGroupNorm(channels, initial_fraction)
        else:
            raise ValueError(normalize)

        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=1, bias=False
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class LatentFiLM(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.affine = nn.Linear(latent_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h, z):
        gain, shift = self.affine(z).chunk(2, dim=1)
        return (
            h * (1 + gain[:, :, None, None])
            + shift[:, :, None, None]
        )


class VAE(nn.Module):
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
        self.fc_decode = nn.Linear(
            self.latent_dim, c3 * 4 * 4
        )

        self.dec0 = ResidualBlock(
            c3, normalize="blend", initial_fraction=0.30
        )

        self.up1_conv = nn.Conv2d(
            c3, c2, 3, padding=1, bias=False
        )
        self.up1_norm = BlendGroupNorm(c2, 0.30)

        self.dec1 = ResidualBlock(
            c2, normalize="blend", initial_fraction=0.30
        )

        self.up2_conv = nn.Conv2d(
            c2, c1, 3, padding=1, bias=False
        )
        self.up2_norm = BlendGroupNorm(c1, 0.10)

        self.dec2 = ResidualBlock(
            c1, normalize="blend", initial_fraction=0.10
        )

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.cond0 = LatentFiLM(c3, self.latent_dim)
        self.cond1 = LatentFiLM(c2, self.latent_dim)
        self.cond2 = LatentFiLM(c1, self.latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def decode(self, z):
        c3 = self.widths[2]

        h = self.fc_decode(z).reshape(
            z.shape[0], c3, 4, 4
        )

        h = self.cond0(self.dec0(h), z)

        h = F.interpolate(
            h, scale_factor=2, mode="nearest"
        )
        h = F.silu(
            self.up1_norm(self.up1_conv(h))
        )
        h = self.cond1(self.dec1(h), z)

        h = F.interpolate(
            h, scale_factor=2, mode="nearest"
        )
        h = F.silu(
            self.up2_norm(self.up2_conv(h))
        )
        h = self.cond2(self.dec2(h), z)

        return self.head(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    