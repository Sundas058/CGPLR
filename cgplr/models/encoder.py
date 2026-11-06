import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ReIDEncoder(nn.Module):
    def __init__(self, imagenet=False):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if imagenet else None)
        net.layer4[0].conv2.stride = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.neck = nn.BatchNorm1d(2048)
        self.neck.bias.requires_grad_(False)
        self.num_features = 2048

    def forward(self, images):
        return F.normalize(self.neck(self.backbone(images).flatten(1)), dim=1)
