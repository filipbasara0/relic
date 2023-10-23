import torch
import torchvision.transforms as transforms

import numpy as np
from torch import nn

np.random.seed(42)


class ContrastiveLearningViewGenerator(object):

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class GaussianBlur(object):

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3,
                                3,
                                kernel_size=(kernel_size, 1),
                                stride=1,
                                padding=0,
                                bias=False,
                                groups=3)
        self.blur_v = nn.Conv2d(3,
                                3,
                                kernel_size=(1, kernel_size),
                                stride=1,
                                padding=0,
                                bias=False,
                                groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h,
                                  self.blur_v)

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def get_relic_aug(image_size):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size,
                                     scale=(0.08, 1.0),
                                     ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(int(image_size * 0.1)),
        transforms.RandomSolarize(127, 0.5),
        transforms.ToTensor()
    ])


def get_relic_aug_inference():
    return transforms.Compose([transforms.ToTensor()])
