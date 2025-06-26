import os

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.linalg import sqrtm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель
model = models.inception_v3(pretrained=True, aux_logits=True).to(device)
model.eval()


# Функция для получения признаков из слоя avgpool
def get_features(images, model, transform, batch_size=32):
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_tensors = [transform(img) for img in batch_imgs]
            batch_tensors = torch.stack(batch_tensors).to(device)
            preds = model(batch_tensors)
            features.append(preds.cpu().numpy())
    return np.concatenate(features, axis=0)


# Трансформ для изображений
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Загрузка изображений
def load_images(folder):
    imgs = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, filename)
            img = Image.open(path).convert("RGB")
            imgs.append(img)
    return imgs


# Пути к папкам
real_folder = "./images/hr"
sr_folder = "./images/sr"

# Загружаем изображения
real_imgs = load_images(real_folder)
sr_imgs = load_images(sr_folder)

# Получаем признаки
real_features = get_features(real_imgs, model, transform)
sr_features = get_features(sr_imgs, model, transform)

# Вычисляем статистики
mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)

mu_sr = np.mean(sr_features, axis=0)
sigma_sr = np.cov(sr_features, rowvar=False)

# Расчет FID
diff = mu_real - mu_sr
covmean, _ = sqrtm(sigma_real @ sigma_sr, disp=False)

if np.iscomplexobj(covmean):
    covmean = covmean.real

fid = diff @ diff + np.trace(sigma_real + sigma_sr - 2 * covmean)
print(f"FID: {fid:.4f}")
