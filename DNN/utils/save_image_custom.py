import os
import numpy as np
from PIL import Image
from argparse import Namespace
from torchvision import transforms
import sys
import itertools
sys.path.append(os.path.expanduser('~/david/master_scripts/image'))
from image_processing import tile

def save_image_custom(inputs=None, outdir=None, labels=None, max_images=128):

    os.makedirs(outdir, exist_ok=True)
    
    # pad view dimension
    inputs = [inputs] if type(inputs) is not list else inputs
    num_views = len(inputs)
    num_images = min(inputs[0].shape[0], max_images)
    outpaths = []
    for i, v in itertools.product(range(num_images), range(num_views)):
        image = inputs[v][i, :, :, :].squeeze()
        image_array = np.array(image.permute(1, 2, 0))
        image_pos = image_array - image_array.min()
        image_scaled = image_pos * (255.0 / image_pos.max())
        image_PIL = Image.fromarray(image_scaled.astype(np.uint8))
        label_string = f'_{labels[int((v*batch_size)+i)]}' if labels else ''
        view_string = f'_view-{v}' if num_views > 1 else ''
        outpath = f'{outdir}/{i:04}{view_string}{label_string}.png'
        image_PIL.save(outpath)
        outpaths.append(outpath)

    colgap, colgapfreq = (8, num_views) if num_views > 1 else (None, None)
    tile(outpaths, f'{outdir}/tiled.png', base_gap=2, colgap=colgap,
         colgapfreq=colgapfreq)
