import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import vgg19
import numpy as np
from PIL import Image


# Генератор
class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.PReLU()
        )

        # Остаточные блоки
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Увеличение разрешения
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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


# Перцептуальные потери на основе VGG19
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:35].eval()
        self.vgg = nn.Sequential(*list(vgg.children())[:35])
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.loss(vgg_input, vgg_target)


# Настройка обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)
vgg_loss = VGGLoss().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixel = nn.MSELoss()


# Пример цикла обучения
def train(epochs, dataloader):
    for epoch in range(epochs):
        for i, (low_res, high_res) in enumerate(dataloader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            # Обучение генератора
            optimizer_G.zero_grad()

            generated = generator(low_res)
            pred_real = discriminator(high_res)
            pred_fake = discriminator(generated.detach())

            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(),
                                     torch.ones_like(pred_fake))
            loss_pixel = criterion_pixel(generated, high_res)
            loss_vgg = vgg_loss(generated, high_res)

            loss_G = 0.001 * loss_GAN + 0.006 * loss_vgg + loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # Обучение дискриминатора
            optimizer_D.zero_grad()

            pred_real = discriminator(high_res)
            pred_fake = discriminator(generated.detach())

            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) * 0.5

            loss_D.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")


# Пример использования
if __name__ == "__main__":
    # Загрузка данных (нужно реализовать свой Dataset)
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = SRDataset("data/train/lr", "data/train/hr", transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Запуск обучения
    train(100, dataloader)