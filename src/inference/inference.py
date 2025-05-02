import torch
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

# Преобразования
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def enhance_image(input_path, output_path):
    # Загрузка и обработка изображения
    lr_img = Image.open(input_path).convert('RGB')
    lr_tensor = transform(lr_img).unsqueeze(0).to(device)

    # Генерация
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)

    # Преобразование обратно в изображение
    sr_tensor = sr_tensor.squeeze().cpu()
    sr_img = ToPILImage()((sr_tensor * 0.5 + 0.5).clamp(0, 1))
    sr_img.save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    enhance_image(args.input, args.output)