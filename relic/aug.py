import torchvision.transforms as transforms

import numpy as np

np.random.seed(42)


class ViewGenerator(object):

    def __init__(self, image_size, num_global_views, num_local_views,
                 global_transform=None, local_transform=None):
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views

        if not global_transform:
            global_transform = get_global_transforms(image_size)
        if not local_transform:
            local_transform = get_local_transforms(image_size)

        self.global_transform = global_transform
        self.local_transform = local_transform

    def __call__(self, x):
        global_views = [self.global_transform(x) for _ in range(self.num_global_views)]
        local_views = [self.local_transform(x) for _ in range(self.num_local_views)]
        return global_views + local_views


def _grayscale_to_rgb(img):
    if img.mode == "L" or img.mode != "RGB":
        return img.convert("RGB")
    return img


def _round_up_to_odd(num):
    return np.ceil(num) // 2 * 2 + 1


def get_global_transforms(image_size):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.Lambda(_grayscale_to_rgb),
        transforms.RandomResizedCrop(image_size, scale=(0.1, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.GaussianBlur(_round_up_to_odd(int(image_size * 0.1))),
        transforms.RandomSolarize(127, 0.2),
        transforms.ToTensor()
    ])


def get_local_transforms(image_size):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.Lambda(_grayscale_to_rgb),
        transforms.RandomResizedCrop(int(image_size*3/7), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.GaussianBlur(_round_up_to_odd(int(image_size * 0.1))),
        transforms.ToTensor()
    ])


def get_inference_transforms(image_size=(96, 96)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.ToTensor()
    ])
