import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from tqdm import tqdm

import os

from src.data.dataset import SRDataset
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.models.vgg_loss import VGGLoss
from src.utils.get_last_checkpoint_name import get_last_checkpoint_name, get_version_number


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

# Настройка обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(10)

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
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state'])
        discriminator.load_state_dict(checkpoint['discriminator_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])

    for epoch in range(initial_epoch, initial_epoch + epochs):
        epoch_pbar = tqdm(dataloader,
                          desc=f"Epoch {epoch}/{initial_epoch + epochs}",
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

    train_dataset = SRDataset(lr_train_dir, hr_train_dir, transform, patch_size=128)
    print(train_dataset.lr_files, 'train_dataset')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    print(train_loader.dataset.__len__(), 'train_loader')
    # Загрузка последнего чекпоинта модели
    last_checkpoint_name=get_last_checkpoint_name()
    last_checkpoint_path=f"checkpoints/{last_checkpoint_name}"
    version_number=get_version_number(last_checkpoint_name)

    train(epochs=10, dataloader=train_loader, initial_epoch=++version_number, checkpoint_path=last_checkpoint_path)