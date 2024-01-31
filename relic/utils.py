import torch
from torchvision.datasets import CIFAR10, STL10

from relic.aug import ViewGenerator
from relic.encoders import resnet18, resnet50, efficientnet_v2_s
from relic.custom_datasets import tiny_imagenet, food101, imagenet1k


def get_dataset(args):
    dataset_name, dataset_path = args.dataset_name, args.dataset_path
    n_global, n_local = args.num_global_views, args.num_local_views
    if dataset_name == "cifar10":
        return CIFAR10(dataset_path,
                       train=True,
                       download=True,
                       transform=ViewGenerator(32, n_global, n_local))
    elif dataset_name == "stl10":
        return STL10(dataset_path,
                     split='unlabeled',
                     download=True,
                     transform=ViewGenerator(96, n_global, n_local))
    elif dataset_name == "tiny_imagenet":
        return tiny_imagenet(transform=ViewGenerator(64, n_global, n_local))
    elif dataset_name == "food101":
        return food101(transform=ViewGenerator(192, n_global, n_local))
    elif dataset_name == "imagenet1k":
        return imagenet1k(transform=ViewGenerator(192, n_global, n_local))

    raise Exception("Invalid dataset name - options are [cifar10, stl10]")


def get_encoder(model_name, modify_model=False):
    if model_name == "resnet18":
        return resnet18(modify_model)
    elif model_name == "resnet50":
        return resnet50(modify_model)
    elif model_name == "efficientnet":
        return efficientnet_v2_s()
    raise Exception(
        "Invalid model name - options are [resnet18, resnet50, efficientnet]")


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
