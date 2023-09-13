from PIL import Image
import os
from torch.utils.data import Dataset
from torch import stack
import natsort
import numpy as np
import glob
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform

        # if list of image paths is submitted, use that
        if len(main_dir[0]) > 1:
            self.all_imgs = main_dir

        # otherwise, get all images in directory
        else:
            self.all_imgs = sorted(glob.glob(f'{main_dir}/*'))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        image = Image.open(self.all_imgs[idx]).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

