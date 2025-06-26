import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


# Перцептуальные потери на основе VGG19
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
