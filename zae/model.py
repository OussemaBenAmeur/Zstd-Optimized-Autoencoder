import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, slope):
        super().__init__()
        self.body = nn.Sequential(
            nn.LeakyReLU(slope), nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(slope), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


def up(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, 4 * cout, 3, padding=1), nn.PixelShuffle(2))


class AutoEncoder(nn.Module):
    def __init__(self, hidden_channels=128, latent_channels=64,
                 downsampling_stages=4, negative_slope=0.2, name="residual_ae"):
        super().__init__()
        if name != "residual_ae" or min(hidden_channels, latent_channels, downsampling_stages) < 1:
            raise ValueError("Invalid model configuration")
        n, c, s = hidden_channels, latent_channels, negative_slope
        self.factor = 2 ** downsampling_stages
        layers, cin = [], 3
        for _ in range(downsampling_stages):
            layers.extend([nn.Conv2d(cin, n, 3, stride=2, padding=1), ResidualBlock(n, s)])
            cin = n
        layers.extend([nn.LeakyReLU(s), nn.Conv2d(n, c, 3, padding=1)])
        self.encoder = nn.Sequential(*layers)
        layers = [nn.Conv2d(c, n, 3, padding=1), ResidualBlock(n, s)]
        for _ in range(downsampling_stages - 1):
            layers.extend([up(n, n), ResidualBlock(n, s)])
        layers.extend([nn.LeakyReLU(s), up(n, 3)])
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        return self.encoder(x - 0.5)

    def decode(self, z):
        return self.decoder(z) + 0.5

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
