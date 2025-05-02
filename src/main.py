import torch
from models.generator import Generator
from models.discriminator import Discriminator
from losses.vgg_loss import VGGLoss
from datasets.dataset import SuperResolutionDataset
from utils.trainer import train

if __name__ == "__main__":
    # Инициализация компонентов
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg_loss = VGGLoss().to(device)

    # Инициализация оптимизаторов
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # Загрузка данных
    dataset = SuperResolutionDataset(...)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

    # Запуск обучения
    train(
        generator=generator,
        discriminator=discriminator,
        vgg_loss=vgg_loss,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        criterion_GAN=torch.nn.BCEWithLogitsLoss(),
        criterion_pixel=torch.nn.MSELoss(),
        epochs=100,
        dataloader=dataloader,
        device=device
    )

    # Сохранение модели
    torch.save(generator.state_dict(), "generator.pth")