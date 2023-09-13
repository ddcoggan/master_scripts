import torch.nn
import torchvision.transforms as transforms
import sys
import os
import shutil
from types import SimpleNamespace
from utils import Occlude


class MultipleViews:

    def __init__(self, transforms):

        self.transforms = transforms
        self.num_views = len(self.transforms)
        
    def __call__(self, input):

        return [self.transforms[v](input) for v in range(self.num_views)]


def get_transforms(D=SimpleNamespace()):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # default transforms
    transforms_train = [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]
    transforms_val =[
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        normalize,
    ]

    # contrastive learning transform
    if hasattr(D, 'transform_type') and D.transform_type == 'contrastive':
        transforms_train = [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ]
        transforms_val = transforms_train.copy()

    # add occlusion
    if hasattr(D, 'Occlusion'):
        transforms_train.insert(3, Occlude(D))
        transforms_val.insert(3, Occlude(D))

    # compose transforms
    transform_train = transforms.Compose(transforms_train)
    transform_val = transforms.Compose(transforms_val)

    # create multiple views
    if D.num_views > 1:

        # for views with different transforms, create list of transforms
        transforms_train = [transforms_train.copy() for _ in range(D.num_views)]
        transforms_val = [transforms_val.copy() for _ in range(D.num_views)]
        for view in range(D.num_views):
            if hasattr(D, 'views_occluded') and view not in D.views_occluded:
                transforms_train[view].pop(3)
                transforms_val[view].pop(3)

        # compose transforms
        transform_train = [transforms.Compose(t) for t in transforms_train]
        transform_val = [transforms.Compose(t) for t in transforms_val]

        # get views
        transform_train = MultipleViews(transform_train)
        transform_val = MultipleViews(transform_val)

    return transform_train, transform_val




