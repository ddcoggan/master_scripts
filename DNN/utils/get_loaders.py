import torch.nn
import torchvision.transforms as transforms
import sys
import os
import os.path as op
import shutil
from types import SimpleNamespace
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .Occlude import Occlude
from .get_transforms import get_transforms

def get_loaders(D,T,num_workers):

    batch_size_adjusted = T.batch_size // D.num_views
    transform_train, transform_val = get_transforms(D)

    path_train = op.expanduser(f'~/Datasets/{D.dataset}/train')
    path_val = op.expanduser(f'~/Datasets/{D.dataset}/val')


    data_train = ImageFolder(path_train, transform=transform_train)
    data_val = ImageFolder(path_val, transform=transform_val)

    loader_train = DataLoader(data_train, batch_size=batch_size_adjusted,
                              shuffle=True, num_workers=num_workers, drop_last=True)
    loader_val = DataLoader(data_val, batch_size=batch_size_adjusted,
                            shuffle=True, num_workers=num_workers)

    return loader_train, loader_val




