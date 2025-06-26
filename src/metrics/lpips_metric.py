import lpips
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm

# Инициализация LPIPS с предобученной моделью (по умолчанию VGG)
loss_fn = lpips.LPIPS(net='vgg', version='0.1')  # 'alex' или 'vgg'
loss_fn.cpu()  # Для GPU. Для CPU: loss_fn.cpu()

# Пути к папкам
hr_dir = './images/hr'
srgan_dir = './images/sr'
results = []

# Получение списка файлов с расширениями .png и .jpg
files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print("Текущая директория:", os.getcwd())
# Обработка каждой пары изображений
for filename in tqdm(files, desc='Calculating LPIPS'):
    hr_path = os.path.join(hr_dir, filename)
    sr_path = os.path.join(srgan_dir, filename)

    # Чтение изображений
    hr_img = cv2.imread(hr_path)
    if hr_img is None:
        print(f"Не удалось прочитать изображение: {hr_path}")
        continue  # пропускаем этот файл

    sr_img = cv2.imread(sr_path)
    if sr_img is None:
        print(f"Не удалось прочитать изображение: {sr_path}")
        continue  # пропускаем этот файл

    # Конвертация BGR в RGB
    hr_img = hr_img[:, :, ::-1]
    sr_img = sr_img[:, :, ::-1]

    # Преобразование в тензоры PyTorch
    hr_tensor = lpips.im2tensor(hr_img).cpu()
    sr_tensor = lpips.im2tensor(sr_img).cpu()

    # Вычисление LPIPS
    with torch.no_grad():
        lpips_value = loss_fn.forward(hr_tensor, sr_tensor)

    results.append(lpips_value.item())

# Расчет среднего, минимального и максимального значения
if results:
    mean_lpips = np.mean(results)
    print(f'Средний LPIPS: {mean_lpips:.4f}')
    print(f'Минимальный: {np.min(results):.4f}, Максимальный: {np.max(results):.4f}')
else:
    print("Нет доступных изображений для расчета LPIPS.")