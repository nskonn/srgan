# Тестирование
def test(generator, dataloader, device):
    generator.eval()
    total_psnr = 0.0

    with torch.no_grad():
        for lr, hr in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = generator(lr)
            psnr = 10 * torch.log10(1 / ((sr - hr) ** 2).mean())
            total_psnr += psnr.item()

    print(f"Average PSNR: {total_psnr / len(dataloader):.2f} dB")


# Запуск тестирования после обучения
val_dataset = SRDataset("data/val/lr", "data/val/hr", transform)
val_loader = DataLoader(val_dataset, batch_size=1)

test(generator, val_loader, device)