import torchvision.transforms as transforms
import sys
import os
import shutil

sys.path.append(f'{os.path.expanduser("~")}/david/masterScripts/DNN')
from utils import AlterImages, multitransforms

def get_transforms(D, T):

    # standard transforms
    
    if not hasattr(D, 'transform') or D.transform == 'standard':
    
        # train
        train_sequence = [
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        # val
        val_sequence = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    elif D.transform == 'alter':

        # transforms allowing for AlterImages functions

        # train
        train_sequence = [
            transforms.ToTensor(),
            transforms.Resize(256),  # resize (smallest edge becomes this length)
            transforms.RandomCrop(256),  # make square
            AlterImages(D, T),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            transforms.RandomCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # val
        val_sequence = [
            transforms.ToTensor(),
            transforms.Resize(256),  # resize (smallest edge becomes this length)
            transforms.CenterCrop(256),  # make square
            AlterImages(D, T),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]


    # for contrastive learning only compose up to the AlterImage stage, as the data loaders cannot handle
    # the splitting of one input to more than one output image during the transforms.
    if hasattr(T, 'learning') and 'contrastive' in T.learning:

        transform_train = transforms.Compose(train_sequence[:3])
        transform_val = transforms.Compose(val_sequence[:3])

    else:

        transform_train = transforms.Compose(train_sequence)
        transform_val = transforms.Compose(val_sequence)

    return transform_train, transform_val


def get_remaining_transform(train_eval):

    # train
    train_sequence = [transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(10),
                      transforms.RandomCrop(224),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    # val
    val_sequence = [transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if train_eval == 'train':
        remaining_sequence = train_sequence
    else:
        remaining_sequence = val_sequence
    remaining_transform = transforms.Compose(remaining_sequence)

    return remaining_transform

