import os
import numpy as np
from PIL import Image
from argparse import Namespace
from torchvision import transforms

def save_image_custom(inputs=None, outdir=None, labels=None, max_images=128):

    os.makedirs(outdir, exist_ok=True)
    
    # pad view dimension
    inputs = [inputs] if type(inputs) is not list else inputs
    num_views = len(inputs)
    n_images = inputs[0].shape[0]
    for v in range(num_views):
        for i in range(min(n_images, max_images)):
            image = inputs[v][i, :, :, :].squeeze()
            image_array = np.array(image.permute(1, 2, 0))
            image_pos = image_array - image_array.min()
            image_scaled = image_pos * (255.0 / image_pos.max())
            image_PIL = Image.fromarray(image_scaled.astype(np.uint8))
            label_string = f'_{labels[int((v*batch_size)+i)]}' if labels else ''
            view_string = f'_view-{v}' if num_views > 1 else ''
            outpath = f'{outdir}/{i:04}{view_string}{label_string}.png'
            image_PIL.save(outpath)
