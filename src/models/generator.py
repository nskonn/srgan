import torch
import torch.nn as nn

from src.models.residual_block import ResidualBlock


# Генератор
class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.final_conv = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.res_blocks(x1)
        x = self.conv2(x) + x1
        x = self.upscale(x)
        x = self.final_conv(x)
        return torch.tanh(x)