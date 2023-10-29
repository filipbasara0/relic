import torchvision.transforms as transforms

import numpy as np

np.random.seed(42)


class ViewGenerator(object):

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def get_relic_aug(image_size):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(int(image_size * 0.1)),
        transforms.RandomSolarize(127, 0.5),
        transforms.ToTensor()
    ])


# TODO: add image_size param and Resize transform
def get_relic_aug_inference():
    return transforms.Compose([transforms.ToTensor()])
