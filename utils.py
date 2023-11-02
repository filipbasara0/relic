import torch
from torchvision.datasets import CIFAR10, STL10

from aug import get_relic_aug, ViewGenerator
from encoders import ConvNext, resnet18, resnet50


def get_dataset(dataset_name, dataset_path):
    if dataset_name == "cifar10":
        return CIFAR10(dataset_path,
                       train=True,
                       download=True,
                       transform=ViewGenerator(get_relic_aug(32), 2))
    elif dataset_name == "stl10":
        return STL10(dataset_path,
                     split='unlabeled',
                     download=True,
                     transform=ViewGenerator(get_relic_aug(96), 2))

    raise Exception("Invalid dataset name - options are [cifar10, stl10]")


def get_encoder(model_name):
    if model_name == "resnet18":
        return resnet18()
    elif model_name == "resnet50":
        return resnet50()
    elif model_name == "convnext":
        return ConvNext(num_channels=3,
                        patch_size=2,
                        layer_dims=[64, 128, 256, 512],
                        depths=[3, 9, 3, 3])
    raise Exception(
        "Invalid model name - options are [resnet18, resnet50, convnext]")


def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.inference_mode
def get_feature_size(encoder):
    """Get the feature size from the encoder using a dummy input."""
    encoder.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = encoder(dummy_input)
    return output.shape[1]
