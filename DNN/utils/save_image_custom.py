import numpy as np
from PIL import Image
from argparse import Namespace
from torchvision import transforms

def save_image_custom(inputs=None, T=Namespace(learning=''), outdir=None, labels=None, max_images=128):

    if 'contrastive' in T.learning:
        repeats = 2
        num_exemplars = inputs.shape[0]/2
    else:
        repeats = 1
        num_exemplars = inputs.shape[0]

    for repeat in range(repeats):
        for i in range(int(min(num_exemplars, max_images))):
            image = inputs[int(num_exemplars*repeat + i), :, :, :].squeeze()
            imagePIL = transforms.ToPILImage()(image)
            if labels:
                outpath = f'{outdir}/{i:04}_{repeat}_{labels[int((repeat*num_exemplars)+i)]}.png'
            else:
                outpath = f'{outdir}/{i:04}_{repeat}.png'
            imagePIL.save(outpath)
