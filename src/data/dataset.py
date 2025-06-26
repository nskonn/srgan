import os
import random

from PIL import Image
from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, patch_size=128):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.patch_size = patch_size  # Размер случайного кропа для обучения

        self.lr_files = sorted(
            [f for f in os.listdir(lr_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        )
        self.hr_files = sorted(
            [f for f in os.listdir(hr_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        )

        assert len(self.lr_files) == len(
            self.hr_files
        ), "Mismatch in LR/HR images count"
        for lr, hr in zip(self.lr_files, self.hr_files):
            assert lr == hr, f"Name mismatch: {lr} vs {hr}"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        # Проверка размеров
        hr_width, hr_height = hr_image.size
        lr_width, lr_height = lr_image.size
        assert (
            hr_width == lr_width * 4 and hr_height == lr_height * 4
        ), f"HR size {hr_image.size} doesn't match LR {lr_image.size}"

        # Случайный кроп фиксированного размера для обучения
        if self.patch_size > 0:
            # Вычисляем допустимые координаты для кропа
            lr_w, lr_h = lr_image.size
            x = random.randint(0, lr_w - self.patch_size)
            y = random.randint(0, lr_h - self.patch_size)

            # Кроп LR изображения
            lr_image = lr_image.crop((x, y, x + self.patch_size, y + self.patch_size))

            # Соответствующий кроп HR изображения (в 4 раза больше)
            hr_x, hr_y = x * 4, y * 4
            hr_patch_size = self.patch_size * 4
            hr_image = hr_image.crop(
                (hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size)
            )

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image
