import torch.nn
import torchvision.transforms.v2 as transforms
import sys
import os
import shutil
from types import SimpleNamespace
from Occlude import Occlude


class MultipleViews:

    def __init__(self, transforms):

        self.transforms = transforms
        self.num_views = len(self.transforms)
        
    def __call__(self, inputs):

        return [self.transforms[v](inputs) for v in range(self.num_views)]


def get_transforms(D):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    # default transforms
    transforms_train = [
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomResizedCrop(D.image_size, antialias=True),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]
    transforms_val = [
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(D.image_size, antialias=True),
        transforms.CenterCrop(D.image_size),
        normalize,
    ]

    # contrastive learning transform
    if hasattr(D, 'transform_type') and D.transform_type == 'contrastive':
        transforms_train = [
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomResizedCrop(D.image_size, scale=(0.8,1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ]
        transforms_val = transforms_train.copy()

    # create list of transforms for each view
    transforms_train = [transforms_train.copy() for _ in range(D.num_views)]
    transforms_val = [transforms_val.copy() for _ in range(D.num_views)]
    
    # add occlusion transform
    if hasattr(D, 'Occlusion'):
        for view in D.Occlusion.views:
            transforms_train[view].insert(4, Occlude(D))
            transforms_val[view].insert(4, Occlude(D))

    # compose transforms
    transforms_train = [transforms.Compose(t) for t in transforms_train]
    transforms_val = [transforms.Compose(t) for t in transforms_val]
    
    # wrap in MultipleViews object
    transforms_train = MultipleViews(transforms_train)
    transforms_val = MultipleViews(transforms_val)
    
    return transforms_train, transforms_val




