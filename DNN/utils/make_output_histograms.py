import glob
import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json
import pickle


def make_output_histograms(activation_file, image_name ='sample'):
    activation = pickle.load(open(activation_file, 'rb'))
    for l in range(len(activation.keys())):
        a = activation[l].cpu()
        a = a.reshape(-1)
        n, bins, patches = plt.hist(x=a, bins=25, color='#0504aa',
                                    alpha=1, rwidth=0.85)
        plt.grid(axis='y', alpha=1)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'layer {l:02d}')
        out_dir = f'{os.path._dir_name(activation_file)}/histograms'
        if not os.path.exists(out_dir):
            os.make_dirs(out_dir)
        file_name = f'{out_dir}/layer{l:02d}.png'
        plt.savefig(file_name)
        plt.close()
