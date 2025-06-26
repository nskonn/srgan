import torch
from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage
from PIL import Image
import os
import argparse

from src.models.generator import Generator

# --- Настройки ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    """Загружает модель генератора."""
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()  # Режим инференса
    return generator

def enhance_single_image(model, input_path, output_path, patch_size=128):
    """Улучшает одно изображение и сохраняет результат."""
    # Загрузка и преобразование изображения
    lr_image = Image.open(input_path).convert("RGB")
    original_size = lr_image.size
    print('1')
    # Трансформации (как при обучении)
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    lr_tensor = transform(lr_image).unsqueeze(0).to(device)  # Добавляем batch-размер

    # Обработка моделью
    with torch.no_grad():
        sr_tensor = model(lr_tensor).squeeze(0).cpu()

    print('2')
    # Денормализация и сохранение
    sr_tensor = (sr_tensor + 1) / 2  # Диапазон [0, 1]
    sr_image = ToPILImage()(sr_tensor)

    print('3')
    # Масштабирование до целевого размера (x4)
    sr_image = sr_image.resize(
        (original_size[0] * 4, original_size[1] * 4),
        Image.BICUBIC
    )
    print('3')
    sr_image.save(output_path)
    print(f"Улучшенное изображение сохранено в: {output_path}")

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Улучшение изображения с помощью SRGAN")
    parser.add_argument("--input", type=str, default="src/inputs/0051x4.png", required=True, help="Путь к входному изображению")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения результата")
    parser.add_argument("--model", type=str, default="checkpoints/generator_final.pth", help="Путь к файлу модели")
    args = parser.parse_args()

    # Проверка путей
    assert os.path.exists(args.input), f"Файл {args.input} не найден!"
    assert os.path.exists(args.model), f"Модель {args.model} не найдена!"

    # Загрузка модели и обработка изображения
    model = load_model(args.model)
    enhance_single_image(model, args.input, args.output)