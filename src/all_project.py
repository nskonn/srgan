import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm

import os

from src.data.dataset import SRDataset
from src.utils.get_last_checkpoint_name import get_last_checkpoint_name


# class SRDataset(Dataset):
#     def __init__(self, lr_dir, hr_dir, transform=None, patch_size=128):
#         self.lr_dir = lr_dir
#         self.hr_dir = hr_dir
#         self.transform = transform
#         self.patch_size = patch_size  # Размер случайного кропа для обучения
#
#         self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
#         self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
#
#         assert len(self.lr_files) == len(self.hr_files), "Mismatch in LR/HR images count"
#         for lr, hr in zip(self.lr_files, self.hr_files):
#             assert lr == hr, f"Name mismatch: {lr} vs {hr}"
#
#     def __len__(self):
#         return len(self.lr_files)
#
#     def __getitem__(self, idx):
#         lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
#         hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
#
#         lr_image = Image.open(lr_path).convert('RGB')
#         hr_image = Image.open(hr_path).convert('RGB')
#
#         # Проверка размеров
#         hr_width, hr_height = hr_image.size
#         lr_width, lr_height = lr_image.size
#         assert hr_width == lr_width * 4 and hr_height == lr_height * 4, \
#             f"HR size {hr_image.size} doesn't match LR {lr_image.size}"
#
#         # Случайный кроп фиксированного размера для обучения
#         if self.patch_size > 0:
#             # Вычисляем допустимые координаты для кропа
#             lr_w, lr_h = lr_image.size
#             x = random.randint(0, lr_w - self.patch_size)
#             y = random.randint(0, lr_h - self.patch_size)
#
#             # Кроп LR изображения
#             lr_image = lr_image.crop((x, y, x + self.patch_size, y + self.patch_size))
#
#             # Соответствующий кроп HR изображения (в 4 раза больше)
#             hr_x, hr_y = x * 4, y * 4
#             hr_patch_size = self.patch_size * 4
#             hr_image = hr_image.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))
#
#         if self.transform:
#             lr_image = self.transform(lr_image)
#             hr_image = self.transform(hr_image)
#
#         return lr_image, hr_image

# Инициализация весов
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


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


# Перцептуальные потери с корректной нормализацией
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
        self.vgg = nn.Sequential(*list(vgg.children())[:35])
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Нормализация для VGG
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, input, target):
        # Денормализация из [-1,1] в [0,1]
        input = (input + 1) / 2
        target = (target + 1) / 2

        # Нормализация для VGG
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.loss(vgg_input, vgg_target)


# Настройка обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация моделей
generator = Generator().to(device)
discriminator = Discriminator().to(device)
initialize_weights(generator)
initialize_weights(discriminator)

vgg_loss = VGGLoss().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixel = nn.MSELoss()


def train(epochs, dataloader, initial_epoch=0, checkpoint_path=None):
    # Загрузка сохранённых состояний (если указан checkpoint)
    # if checkpoint_path:
    #     checkpoint = torch.load(checkpoint_path)
    #     generator.load_state_dict(checkpoint['generator_state'])
    #     discriminator.load_state_dict(checkpoint['discriminator_state'])
    #     optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
    #     optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])

    for epoch in range(initial_epoch, initial_epoch + epochs):
        epoch_pbar = tqdm(dataloader,
                          desc=f"Epoch {epoch + 1}/{initial_epoch + epochs}",
                          leave=False)

        for i, (low_res, high_res) in enumerate(dataloader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            epoch_pbar.update(10)

            # Обучение генератора
            optimizer_G.zero_grad()
            generated = generator(low_res)

            # GAN loss
            pred_real = discriminator(high_res)
            pred_fake = discriminator(generated)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(), torch.ones_like(pred_fake))

            # Pixel and VGG loss
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

            # Обновляем прогресс-бар
            epoch_pbar.set_postfix({
                "Loss D": f"{loss_D.item():.4f}",
                "Loss G": f"{loss_G.item():.4f}",
                "G-Pixel": f"{loss_pixel.item():.4f}",
                "G-VGG": f"{loss_vgg.item():.4f}"
            })
            epoch_pbar.close()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{initial_epoch + epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")

        # Сохранение модели
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state': generator.state_dict(),
                'discriminator_state': discriminator.state_dict(),
                'optimizer_G_state': optimizer_G.state_dict(),
                'optimizer_D_state': optimizer_D.state_dict(),
            }, f"checkpoints/checkpoint_{epoch}.pth")


if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Проверка путей
    lr_train_dir = os.path.expanduser("~/Desktop/DIV2K_train_LR_bicubic/X4")
    hr_train_dir = os.path.expanduser("~/Desktop/DIV2K_train_HR")

    assert os.path.isdir(lr_train_dir), f"LR directory not found: {lr_train_dir}"
    assert os.path.isdir(hr_train_dir), f"HR directory not found: {hr_train_dir}"

    torch.set_num_threads(10)

    # Используем patch_size=128 для случайных кропов 128x128 из LR и 512x512 из HR
    train_dataset = SRDataset(lr_train_dir, hr_train_dir, transform, patch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)

    # Загрузка последнего инференса модели
    last_checkpoint_path=get_last_checkpoint_version()
    print(f"Last checkpoint path: {last_checkpoint_path}")

    # last_checkpoint_path=f"checkpoints/checkpoint_{141}.pth"

    train(epochs=10, dataloader=train_loader, initial_epoch=141, checkpoint_path=last_checkpoint_path)