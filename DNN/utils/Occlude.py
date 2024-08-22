import numpy as np
import os
import os.path as op
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import math
import itertools
from types import SimpleNamespace
import torchvision.transforms.functional as F


class Occlude:

    """
    This class is used to augment images with occlusion. It can be used in
    the same way as any torch vision transform, and is compatible with the
    Compose class. The occluder images are loaded from disk and a separate
    transform is applied before they are superimposed on the input image.
    Args:
        D: A SimpleNamespace object containing the dataset configuration.
        This must contain the following attributes:
            image_size: The size of the input images
            Occlusion: A SimpleNamespace object containing the occlusion
                configuration. This must contain the following attributes:
                    form: The form of the occluder as a string, or list
                        thereof. Strings must match one of the directory names
                        in the occlusion dataset, e.g. 'mudSplash'.
                    probability: The probability (0. - 1.) that the occlusion
                        transform is applied to an image, e.g 0.8 will
                        occlude ~80% of the inputs.
                    visibility: The approximate proportion of the image that is
                        to remain visible, i.e. 1. - occluder coverage,
                        or list thereof.
                    color: The color of the occluder, or list thereof. This can
                        be a tuple array of RGB values (0-255) or 'random' to
                        select random RGB values for each occluder. This
                        attibute is ignored for occluder forms with texture,
                        e.g. 'naturalTextured'.
                Note: if a list is submitted, the occluder for each input image
                will be configured by sampling randomly from the list.
        preload: If 'paths', occluder images are loaded from disk at runtime.
            If 'tensors', occluder images are loaded from memory at runtime.
            The 'tensors' option is currently in development and is not
            recommended to use unless you want to complete development and
            test it yourself.
    """

    def __init__(self, D=None, preload='paths'):

        self.D = D
        self.preload = preload

        O = self.D.Occlusion
        occluder_dir = op.expanduser('~/Datasets/occluders')

        # transform for occluder only
        self.occluder_transform = transforms.Compose([
            transforms.Resize(D.image_size, Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(D.image_size)])

        # ensure occluders and visibilities are lists
        occ_forms = [O.form] if isinstance(O.form, str) else O.form
        visibilities = [O.visibility] if \
            type(O.visibility) in [str, int, float] else O.visibility

        # specify occluders at instantiation for better training speed
        occ_dirs = []
        for of, vis in itertools.product(occ_forms, visibilities):
            if of != 'unoccluded':

                # for occluders where visibility is not controllable
                if of.startswith('natural') and 'Cropped' not in of:
                    occ_dirs += \
                        glob.glob(f'{occluder_dir}/{of}')

                # for occluders where visibility is controllable
                else:
                    if vis == 'random':
                        occ_dirs += glob.glob(f'{occluder_dir}/{of}/*')
                    else:
                        cov = round((1 - vis) * 100)  # dirs represent % coverage
                        occ_dirs += glob.glob(f'{occluder_dir}/{of}/{cov}')

        # track which occluders should be presented with texture
        self.textured = []

        # method 1: store paths to occluder images and load at runtime
        if preload == 'paths':
            self.occluders = []
            for occ_dir in occ_dirs:
                image_paths = glob.glob(f'{occ_dir}/*.png')
                self.occluders += image_paths
                textured = 'Textured' in occ_dir
                self.textured += [textured] * len(image_paths)

        # method 2: preload occluder tensors for even more speed
        else:
            tensor_paths = [f'{occ_dir}/occluders.pt' for occ_dir in occ_dirs]
            textured = ['Textured' in occ_dir for occ_dir in occ_dirs]
            self.occluders = torch.empty(0, 4, D.image_size, D.image_size)
            for tensor_path in tensor_paths:
                occluder = torch.load(tensor_path)
                if occluder.ndim == 3:
                    occluder = torch.stack([occluder] * 4, dim=1)
                self.occluders = torch.concat([self.occluders, occluder], dim=0)
                self.textured += [textured] * occluder.shape[0]


    def __call__(self, image):

        O = self.D.Occlusion

        if O.form != 'unoccluded' and torch.rand(1) < O.probability:

            # method 1: load occluder image from disk
            if self.preload == 'paths':
                occluder_idx = torch.randint(len(self.occluders), (1,))
                occluder_path = self.occluders[occluder_idx]
                occluder = transforms.PILToTensor()(
                    Image.open(occluder_path))

            # method 2: load occluder tensor from memory
            else:
                occluder_idx = torch.randint(self.occluders.shape[0], (1,))
                occluder = self.occluders[occluder_idx].squeeze()


            # ensure range [0,1]
            if occluder.max() > 1.1:
                occluder = occluder / 255
            occluder = occluder.clip(0,1)


            # set occluder color unless texture is used
            if not self.textured[occluder_idx]:

                # if multiple colors requested, select one at random
                if type(O.color) == list:
                    fill_col = O.color[torch.randint(len(O.color),(1,))]
                else:
                    fill_col = O.color

                # if color is specified as RGB, convert to tensor and normalise
                if fill_col == 'random':
                    fill_rgb = torch.rand((3,))
                else:
                    fill_rgb = torch.tensor(fill_col)
                    if max(fill_rgb) > 1:
                        fill_rgb /= 255

                # colorize
                occluder[:3] += fill_rgb[:, None, None]


            # transform
            occluder = self.occluder_transform(occluder)

            # get object and occluder RGB masks from occluder alpha channel 
            occluded_pixels = torch.tile(occluder[3, :, :], dims=(3, 1, 1))
            visible_pixels = 1 - occluded_pixels
            
            # zero occluded pixels in object and visible pixels in occluder
            image *= visible_pixels
            occluder[:3] *= occluded_pixels  # need for untextured

            # replace occluded pixels with occluder (dropping alpha channel)
            image += occluder[:3]

        return image

    """
    # Code for viewing tensor as image
    import matplotlib.pyplot as plt
    plt.imshow(image.permute(1,2,0))
    plt.show()
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
        
        
        
