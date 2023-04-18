import numpy as np
from PIL import Image
from argparse import Namespace

def save_image_custom(inputs=None, t=Namespace(learning=''), outdir=None, labels=None, max_images=128):

    if 'contrastive' in t.learning:
        repeats = 2
        num_exemplars = inputs.shape[0]/2
    else:
        repeats = 1
        num_exemplars = inputs.shape[0]

    for repeat in range(repeats):
        for i in range(int(min(num_exemplars, max_images))):
            image = inputs[int(num_exemplars*repeat + i), :, :, :].squeeze()
            imageArray = np.array(image.permute(1, 2, 0))
            imagePos = imageArray - imageArray.min()
            imageScaled = imagePos * (255.0 / imagePos.max())
            imagePIL = Image.fromarray(imageScaled.astype(np.uint8))
            if labels:
                outpath = f'{outdir}/{i:04}_{repeat}_{labels[int((repeat*num_exemplars)+i)]}.png'
            else:
                outpath = f'{outdir}/{i:04}_{repeat}.png'
            imagePIL.save(outpath)
