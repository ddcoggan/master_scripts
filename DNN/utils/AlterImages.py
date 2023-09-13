import numpy as np
import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import itertools
from types import SimpleNamespace
import torchvision.transforms.functional as F
from tensordict.tensordict import TensorDict


class AlterImages:

    def __init__(self, D=None):
        self.D = D

        self.multiview = hasattr(D, 'num_views') and D.num_views > 1

        self.alter = False
        alterations = ['Occlusion', 'Blur', 'Noise', 'greyscale']
        for a in alterations:
            if hasattr(D, a):
                self.alter = True


    def __call__(self, inputs):

        if not self.multiview and not self.alter:
            return inputs
        else:

            # if single image (3-D tensor) pad 4th dimension
            if len(inputs.shape) == 3:
                inputs = inputs[None, :]

            # first dimension then always represents number of images
            num_images = inputs.shape[0]

            # split into views
            if self.multiview:
                num_views = self.D.num_views
                views = torch.tile(inputs, dims=(1, num_views, 1, 1, 1))
            else:
                num_views = 1
                views = torch.tile(inputs, dims=(num_views, 1, 1, 1))
                views = views[None, :] # pad image dimension


            # apply transforms

            # if contrastive learning using standard contrastive transform
            if self.multiview and not self.alter:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(
                        brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                                           p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                ])
                contrastive_views = torch.empty(views.shape[:-2] + (224, 224))
                for i, v in itertools.product(np.arange(num_images), np.arange(
                        num_views)):
                    contrastive_views[i, v, :] = transform(views[i, v, :])
                views = contrastive_views.squeeze()

            # if using custom transforms
            if hasattr(self.D, 'Blur'):
                views = blur_image(views, self.D)
            if hasattr(self.D, 'Noise'):
                views = add_noise(views, self.D)
            if hasattr(self.D, 'Occlusion'):
                views = occlude_image(views, self.D)
            if hasattr(self.D, 'greyscale') and self.D.greyscale:
                views = greyscale_image(views)

            return views

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
def occlude_image(inputs, D):

    """Adds occluders to image.

        Arguments:
            inputs (tensor):
            o = occlusion params (Namespace): type (string or list): type of occlusion to apply.
                                      visibility (float, range 0:1): proportion of image to remain visible
                                      colour (tuple of length 3 or list thereof, range 0:255): RGB colour for occluded pixels
            seed: 

        Returns:
            occluded image (tensor)"""


    O = D.Occlusion
    occluder_dir = os.path.expanduser('~/Datasets/occluders')

    num_images, num_views, C, H, W = inputs.shape
    outputs = torch.empty(num_images, num_views, C, 224, 224)

    """
    # preload all occluder tensors
    all_occluders = TensorDict({}, batch_size=[1])
    if not isinstance(O.type, list):
        O.type = [O.type]
    if not isinstance(O.visibility, list):
        O.visibility = [O.visibility]
    for ot in O.type:
        all_occluders[ot] = TensorDict({}, batch_size=[1])
        for ov in O.visibility:
            if ov is None:
                occluders_path = f'{occluder_dir}/{ot}/{round(ov * 100)}/occluders.pt'
                all_occluders[ot] = torch.load(occluders_path)
            elif ov < 1:
                occluders_path = f'{occluder_dir}/{ot}/{round(ov * 100)}/occluders.pt'
                all_occluders[ot][1-ov] = torch.load(occluders_path)
    """


    # transform for occluder only
    occluder_transform = transforms.Compose([
        transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(1.25, 1.25)), # scale to remove edge artifacts
        ])

    # transform for view prior to occlusion
    view_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ])

    # transform for occluded image
    final_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        ])

    # loop through images, views, applying occlusion
    for i in range(num_images):

        views = inputs[i]


        # prop_occluded is applied image-wise
        occlude_views = torch.rand(1) < O.prop_occluded

        # ensure first occluded view has new occluder
        occ_configured = False

        for v in range(num_views):

            image = views[v]
            if num_views > 1:
                image = view_transform(image)

            # determine whether to occlude this view
            occlude_view = v in D.views_altered if hasattr(D, 'views_altered') else True
            if not occlude_view or not occlude_views:
                outputs[i, v] = image
            else:

                # get occluder parameters
                if not occ_configured or D.view_resample:  # use same occluder for all views unless resample

                    # image visibility
                    # note: directories are labelled by occluder coverage (1 - visibility)
                    if O.visibility is None:  # in case occluder type has no specified visibilities
                        this_coverage = None
                        coverage_path = ''
                    else:
                        if type(O.visibility) == list:
                            this_coverage = 1 - O.visibility[
                                torch.randint(len(O.visibility), (1,))]  # if multiple, select one at random
                        else:
                            this_coverage = 1 - O.visibility
                        coverage_path = f'{round(this_coverage * 100)}/'

                    # occluder type
                    if type(O.type) == list:
                        this_occ_type = O.type[torch.randint(len(O.type), (1,))]  # if multiple, select one at random
                    else:
                        this_occ_type = O.type

                    # occluder instance
                    """
                    occs = all_occluders[this_occ_type][this_coverage]
                    occluder_image = occs[torch.randint(occs.shape[0], (1,))]
                    """
                    occluder_paths = glob.glob(f'{occluder_dir}/{this_occ_type}/{coverage_path}*.png')
                    occluder_path = occluder_paths[torch.randint(len(occluder_paths), (1,))]
                    occluder_image = transforms.PILToTensor()(Image.open(occluder_path).convert('RGBA').resize((224,224), resample=Image.BILINEAR))


                    # occluder colour
                    if 'naturalTextured' in this_occ_type:
                        # keep original colour
                        occluder_base = occluder_image / 255
                    else:
                        # use custom colour
                        if type(O.colour) == list:
                            fill_col = O.colour[torch.randint(len(O.colour), (1,))]  # if multiple, select one at random
                        else:
                            fill_col = O.colour
                        if isinstance(fill_col, tuple):
                            fill_col = torch.tensor(fill_col)
                            if max(fill_col) > 1:
                                fill_col = fill_col / 255
                        elif fill_col == 'random':
                            fill_col = torch.rand((3,))

                        fill_col = torch.concat([fill_col, torch.tensor((1,))]) # add alpha
                        occluder_base = (occluder_image[3] / 255) * fill_col[:,None,None].expand((4,224,224))

                    occ_configured = True

                # view-specific occluder transform
                occluder = occluder_transform(occluder_base)

                # occluder mask is occluder alpha channel tiled across RGB channels
                occluder_mask = torch.tile(occluder[3,:,:], dims=(3, 1, 1))

                # remove occluded pixels from image
                image *= (1 - occluder_mask)

                # replace occluded pixels with fill colour (dropping alpha channel)
                image += occluder[:3]

                # perform final transform on occluded image
                image = final_transform(image)

                outputs[i,v] = image

    outputs = outputs.squeeze() # remove leading dimensions if one output image

    return outputs



def add_noise(image, D):

    N = D.noise
    noisy_image = torch.zeros_like(image)

    if type(n.ssnr) is list:
        ssnr = random.choice(N.ssnr)
    else:
        ssnr = N.ssnr

    if n.type == 'gaussian': # Gaussian noise
        sigma = (1 - ssnr) / 2 / 3
        signal = (image-0.5) * ssnr + 0.5
        noise = np.tile(np.random.normal(0, sigma, (1, image.size(2), image.size(3))), (image.size(1), 1, 1))
        noise = torch.from_numpy(noise).float().to(image.device)
        noisy_image = signal + noise
        noisy_image[noisy_image > 1] = 1
        noisy_image[noisy_image < 0] = 0
        noisy_image = normalize(noisy_image)
        noisy_image[i] = noisy_image

    elif n.type == 'fourier':

        image_fft = np.fft.fft2(np.mean(image.cpu().detach().np(), axis=0))
        image_fft_phase = np.angle(image_fft)
        np.random.shuffle(image_fft_phase.flat)
        image_fft_shuffled = np.multiply(image_fft_avg_mag, np.exp(1j * image_fft_phase))
        image_recon = abs(np.fft.ifft2(image_fft_shuffled))
        image_recon = (image_recon - np.min(image_recon)) / (np.max(image_recon) - np.min(image_recon))

        signal = (image - 0.5) * ssnr + 0.5
        noise = np.tile((image_recon - 0.5) * (1 - ssnr), (image.size(1), 1, 1))
        noise = torch.from_np(noise).float().to(image.device)
        noisy_image = signal + noise
        noisy_image[noisy_image > 1] = 1
        noisy_image[noisy_image < 0] = 0
        noisy_image = normalize(noisy_image)

    noisy_image = noisy_image.repeat(1, 3, 1, 1) # RGB
    return noisy_image


def blur_image(image, D):

    B = D.blur
    weights = np.asarray(B.weights).astype('float64')
    weights = weights / np.sum(weights)
    sigma = random.choice(B.sigmas, 1, p=weights)[0]
    kernel_size = 2 * math.ceil(2.0 * sigma) + 1

    if sigma == 0:
        blurred_image = image
    else:
        blurred_image = F.gaussian_blur(image, kernel_size, [sigma, sigma])

    return blurred_image

def greyscale_image(inputs):
    grey_image = transforms.Grayscale()(inputs)
    return grey_image
