import numpy as np
import os
import glob
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
from types import SimpleNamespace
import torchvision.transforms.functional as F

class AlterImages:

    def __init__(self, D=None, T=None):
        self.D = D
        self.T = T

    def __call__(self, pic):
        if hasattr(self.D, 'Blur'):
            pic = blur_image(pic, self.D)
        if hasattr(self.D, 'Noise'):
            pic = add_noise(pic, self.D)
        if hasattr(self.D, 'Occlusion'):
            pic = occlude_image(pic, self.D, self.T)
        if hasattr(self.D, 'greyscale') and self.D.greyscale:
            pic = greyscale_image(pic)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
def occlude_image(inputs, D, T=None):

    """Adds occluders to image.

        Arguments:
            image (tensor):
            o = occlusion params (Namespace): type (string or list): type of occlusion to apply.
                                      visibility (float, range 0:1): proportion of image to remain visible
                                      colour (tuple of length 3 or list thereof, range 0:255): RGB colour for occluded pixels
            seed: 

        Returns:
            occluded image (tensor)"""

    o = D.Occlusion
    occluder_dir = f'{os.path.expanduser("~")}/Datasets/occluders'

    # output under contrastive learning is twice the size
    if 'contrastive' in T.learning:
        outputs = torch.empty_like(torch.tile(inputs, dims=(2,1,1,1)))
    else:
        outputs = torch.empty_like(inputs)

    # pad 4th dimension to allow image loop over a single image
    if len(inputs.shape) == 3:
        inputs = inputs[None, :]

    num_images = inputs.shape[0]

    for i in range(num_images):

        image = torch.squeeze(inputs[i,:,:,:])

        H,W = image.shape[1:3]

        # get visibility
        if o.visibility is None: # in case occluder type has no specified visibilities
            this_coverage = None
            coverage_path = ''
        else:
            if type(o.visibility) == list:
                this_coverage = 1 - random.choice(o.visibility)
            else:
                this_coverage = 1 - o.visibility
            coverage_path = f'{round(this_coverage * 100)}/'

        # get fill colour
        if type(o.colour) == list:
            fill_col = torch.tensor(random.choice(o.colour))
        elif type(o.colour) == tuple:
            fill_col = torch.tensor(o.colour)
        elif o.colour == 'random':
            fill_col = torch.randint(255,(3,))

        # if no occlusion desired
        if this_coverage == 0 or np.random.uniform() > o.prop_occluded:

            if 'contrastive' in T.learning:

                # first image is unaltered
                outputs[i, :, :, :] = image

                # second image is translated slightly to avoid identical image portions during val
                image_PIL = transforms.ToPILImage()(image).convert('RGBA')
                x_shift = np.random.randint(41) - 20
                y_shift = np.random.randint(41) - 20
                image_PIL = image_PIL.rotate(0, translate=(x_shift, y_shift)) # doesn't actually rotate, just translates
                translated_image = torch.tensor(np.array(image_PIL.convert('RGB'))).permute(2, 0, 1) / 255
                outputs[num_images + i, :, :, :] = translated_image

            else:

                outputs = image


        # if occlusion desired
        else:

            image_PIL = transforms.ToPILImage()(image).convert('RGBA')

            occluders_PIL = []

            # get occlusion image
            if type(o.type) == list:
                this_occ_type = random.choice(o.type)
            else:
                this_occ_type = o.type
            if this_occ_type in ['naturalTextured', 'naturalTextured1', 'naturalTextured2']:
                occluder_paths = glob.glob(f'{occluder_dir}/{this_occ_type}/*.png')
            else:
                occluder_paths = glob.glob(f'{occluder_dir}/{this_occ_type}/{coverage_path}*.png')
            occluder_path = random.choice(occluder_paths)
            occluder_PIL = Image.open(occluder_path).convert('RGBA')
            rotation = np.random.randint(21) - 10 # randomly rotate between +- 10 degrees
            occluders_PIL.append(occluder_PIL.rotate(rotation))

            # if contrastive learning, get second occluder
            if 'contrastive' in T.learning:

                # if second occluder is a different random occluder
                if D.contrast == 'random_occluder':
                    if type(o.type) == list:
                        this_occ_type = random.choice(o.type)
                    else:
                        this_occ_type = o.type
                    if this_occ_type in ['naturalTextured', 'naturalTextured1', 'naturalTextured2']:
                        occluder_paths = glob.glob(f'{occluder_dir}/{this_occ_type}/*.png')
                    else:
                        occluder_paths = glob.glob(f'{occluder_dir}/{this_occ_type}/{coverage_path}*.png')
                    occluder_path = random.choice(occluder_paths)
                    occluder_PIL = Image.open(occluder_path).convert('RGBA')
                    rotation = np.random.randint(21) - 10  # randomly rotate occluder +- 10 degrees
                    occluders_PIL.append(occluder_PIL.rotate(rotation))

                # if second image is translated version of same occluder
                if D.contrast == 'occluder_translate':
                    x_shift = np.random.randint(41) - 20
                    y_shift = np.random.randint(41) - 20
                    occluders_PIL.append(occluders_PIL[0].copy())
                    occluders_PIL[1] = occluders_PIL[1].rotate(0, translate=(x_shift, y_shift))

            for x, occluder_PIL in enumerate(occluders_PIL):

                occluded_image_PIL = image_PIL.copy()

                # if contrastive learning, translate second image prior to occlusion
                if x == 1:
                    x_shift = np.random.randint(41) - 20
                    y_shift = np.random.randint(41) - 20
                    occluded_image_PIL = occluded_image_PIL.rotate(0, translate=(x_shift, y_shift))

                # ensure image and occluder are same size
                if image_PIL.size != occluder_PIL.size:
                    occluder_PIL = occluder_PIL.resize(image_PIL.size)

                # if occluder is textured, paste occluder over image
                if 'naturalTextured' in this_occ_type:
                    occluded_image_PIL.paste(occluder_PIL, (0, 0), occluder_PIL)
                    occluded_image = torch.tensor(np.array(occluded_image_PIL.convert('RGB')))
                    if 'contrastive' in T.learning:
                        outputs[x*num_images + i, :, :, :] = occluded_image.permute(2, 0, 1) / 255
                    else:
                        outputs = occluded_image.permute(2,0,1)/255

                # if occluder is untextured, set colour and paste over image
                else:
                    # empty final occluder image
                    occluder_coloured = np.zeros((H, W, 4), dtype=np.uint8)

                    # fill first 3 channels with fill colour
                    occ_colour_PIL = Image.new(color=tuple(fill_col),mode='RGB',size=(H,W))
                    occluder_coloured[:, :, :3] = np.array(occ_colour_PIL)

                    # fill last channel with alpha layer of occluder
                    occluderAlpha = torch.tensor(np.array(occluder_PIL))[:, :, 3]  # load image, put in tensor
                    occluder_coloured[:,:,3] = occluderAlpha

                    # make image
                    occluder_coloured_PIL = Image.fromarray(occluder_coloured, mode='RGBA')

                    # paste occluder over image
                    occluded_image_PIL.paste(occluder_coloured_PIL, (0,0), occluder_coloured_PIL)
                    occluded_image = torch.tensor(np.array(occluded_image_PIL.convert('RGB')))

                    if 'contrastive' in T.learning:
                        outputs[x*num_images + i, :, :, :] = occluded_image.float().permute(2,0,1)/255
                    else:
                        outputs = occluded_image.float().permute(2,0,1)/255

    return outputs



def add_noise(image, d):

    n = D.noise
    noised_image = torch.zeros_like(image)

    if type(n.ssnr) is list:
        ssnr = random.choice(n.ssnr)
    else:
        ssnr = n.ssnr

    if n.type == 'gaussian': # Gaussian noise
        sigma = (1 - ssnr) / 2 / 3
        signal = (image-0.5) * ssnr + 0.5
        noise = np.tile(np.random.normal(0, sigma, (1, image.size(2), image.size(3))), (image.size(1), 1, 1))
        noise = torch.from_numpy(noise).float().to(image.device)
        noised_image = signal + noise
        noised_image[noised_image > 1] = 1
        noised_image[noised_image < 0] = 0
        noised_image = normalize(noised_image)
        noised_image[i] = noised_image

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
        noised_image = signal + noise
        noised_image[noised_image > 1] = 1
        noised_image[noised_image < 0] = 0
        noised_image = normalize(noised_image)

    noised_image = noised_image.repeat(1, 3, 1, 1) # RGB
    return noised_image


def blur_image(image, d):

    b = D.blur
    weights = np.asarray(b.weights).astype('float64')
    weights = weights / np.sum(weights)
    sigma = random.choice(b.sigmas, 1, p=weights)[0]
    kernel_size = 2 * math.ceil(2.0 * sigma) + 1

    if sigma == 0:
        blurred_image = image
    else:
        blurred_image = F.gaussian_blur(image, kernel_size, [sigma, sigma])

    return blurred_image

def greyscale_image(inputs):
    if len(inputs.shape) == 3:
        greyscale_image = torch.tile(torch.mean(inputs, axis=0,keepdim=True), dims=(3,1,1))
    elif len(inputs.shape) == 4:
        greyscale_image = torch.tile(torch.mean(inputs, axis=1,keepdim=True), dims=(1,3,1,1))
    return greyscale_image
