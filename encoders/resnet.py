import torch.nn as nn
from torchvision import models


class Squeeze(nn.Module):

    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


def resnet18():
    resnet = models.resnet18(weights=None)

    modules = list(resnet.children())[:-1]
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(Squeeze())

    resnet_feature_extractor = nn.Sequential(*modules)
    return resnet_feature_extractor