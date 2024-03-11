import torch
import torch.nn as nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self, layer=35):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features.eval()
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer])

    def forward(self, X, Y):
        phi_x = self.feature_extractor(X)
        phi_y = self.feature_extractor(Y)
        loss = ((phi_x - phi_y)**2).mean()
        return loss
