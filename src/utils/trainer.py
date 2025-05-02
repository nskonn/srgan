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
