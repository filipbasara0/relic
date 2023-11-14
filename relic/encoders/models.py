import torch.nn as nn
from torchvision import models


class Squeeze(nn.Module):

    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


def _prep_encoder(model):
    modules = list(model.children())[:-1]
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(Squeeze())

    return nn.Sequential(*modules)


def resnet18():
    resnet = models.resnet18(weights=None)
    return _prep_encoder(resnet)


def resnet50():
    resnet = models.resnet50(weights=None)
    return _prep_encoder(resnet)


def efficientnet_v2_s():
    model = models.efficientnet_v2_s(weights=None)
    return _prep_encoder(model)