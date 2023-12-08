import sys
import pickle
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import numpy as np
import glob
import os
import os.path as op
from PIL import Image
import datetime
import torch
torch.backends.cudnn.benchmark = False
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace
import math
import pprint
import torch.nn.functional as F
import shutil
import pickle as pkl


sys.path.append(os.path.expanduser('~/david/master_scripts/DNN'))
from utils.accuracy import accuracy


def test_model(CFG):  # model params, dataset params, training params

    torch.no_grad()

    # unpack config
    M, D, T = CFG.M, CFG.D, CFG.T

    # get model architecture from model dir
    CFG_orig = pkl.load(open(f'{M.model_dir}/config.pkl', 'rb'))

    ### CALCULATE CONFIGURATION PARAMETERS ###

    # configure hardware
    from utils import configure_hardware
    T, device = configure_hardware(T)

    # load model
    if not hasattr(M, 'model'):
        from utils import get_model
        model = get_model(CFG_orig.M)
    else:
        model = M.model
    model.eval()

    # calculate optimal batch size
    if not hasattr(T, 'batch_size'):
        from utils import calculate_batch_size
        T = calculate_batch_size(model, T, device)
        print(f'optimal batch size calculated at {T.batch_size}')

    ### image processing ###

    from utils import get_transforms
    _, transform = get_transforms(D)
    val_path = op.expanduser(f'~/Datasets/{D.dataset}/val')
    val_data = ImageFolder(val_path, transform=transform)
    if hasattr(D, 'class_subset'):
        from torch.utils.data import Subset
        idxs = [i for i, image_data in enumerate(val_data.imgs) if image_data[1] in D.class_subset]
        val_data = Subset(val_data, idxs)
    loader = DataLoader(val_data, batch_size=T.batch_size, shuffle=True, num_workers=T.workers)

    # loss functions
    if 'classification' in T.learning:
        loss_class = nn.CrossEntropyLoss().to(device)  # send to device before constructing optimizer
    if 'contrastive' in T.learning:
        from utils import ContrastiveLoss
        loss_contr = ContrastiveLoss().to(device)  # can do both supervised and unsupervised contrastive loss


    ### performance metrics ###

    metrics = []
    if 'classification' in T.learning:
        metrics += ['acc1', 'acc5', 'loss_class']
    if 'contrastive' in T.learning:
        metrics += ['loss_contr']
    stats = {}
    for metric in metrics:
        stats[metric] = 0

    # put model on device
    if T.nGPUs > 1:
        model = nn.DataParallel(model)
    model.to(device)
    #print(model)

    # load model parameters
    if not hasattr(CFG.M, 'params_loaded') or CFG.M.params_loaded is False:
        print('loading parameters...')
        from utils import load_params
        model = load_params(M.params_path, model=model)


    ### TEST ###

    # loop through batches
    with tqdm(loader, unit=f"batch({T.batch_size})") as tepoch:

        for batch, (inputs, targets) in enumerate(tepoch):

            tepoch.set_description(
                f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Testing')

            # initialize performance stats for this batch
            batch_stats = {}

            # for contrastive learning, alter images, perform remaining image transform
            if 'contrastive' in T.learning:
                from utils import AlterImages, get_remaining_transform
                inputs = AlterImages(D, T)(inputs)
                remaining_transform = get_remaining_transform('eval')
                inputs = remaining_transform(inputs)

            # put inputs on device
            non_blocking = 'contrastive' in T.learning
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            # pass through model
            outputs = model(inputs)

            # separate by classification/contrastive
            if hasattr('M', 'out_channels') and M.out_channels == 2:
                outputs_class = outputs[:, :, 0]
                outputs_contr = outputs[:, :, 1]
                if 'contrastive' in T.learning:
                    targets_class = torch.cat([targets, targets], dim=0)
                else:
                    targets_class = targets
            elif T.learning == 'supervised_classification':
                outputs_class = outputs
                targets_class = targets
            else:
                outputs_contr = outputs

            # classification accuracy and loss
            if 'classification' in T.learning:

                # subset the data before getting accuracy
                if hasattr(D, 'class_subset'):
                    outputs_subset = outputs_class[:, D.class_subset].detach().cpu()
                    targets_subset = torch.tensor([D.class_subset.index(target) for target in targets_class])
                else:
                    outputs_subset = outputs_class
                    targets_subset = targets_class

                batch_stats['acc1'], batch_stats['acc5'] = [x.detach().cpu().item() for x in accuracy(outputs_subset, targets_subset, (1, 5))]
                stats['acc1'] = ((stats['acc1'] * batch) + batch_stats['acc1']) / (batch + 1)
                stats['acc5'] = ((stats['acc5'] * batch) + batch_stats['acc5']) / (batch + 1)
                batch_stats['loss_class'] = loss_class(outputs_class, targets_class).detach().cpu().item()
                stats['loss_class'] = ((stats['loss_class'] * batch) + batch_stats['loss_class']) / (batch + 1)

            # contrastive accuracy and loss
            if 'contrastive' in T.learning:
                outputs_contr = F.normalize(outputs_contr, p=2, dim=1)  # make output vector unit length (l2 norm)
                f1, f2 = torch.split(outputs_contr, [targets.shape[0], targets.shape[0]], dim=0)  # separate outputs from different loaders
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # combine along new (contrastive) dimension
                if 'unsupervised_contrastive' in T.learning:
                    loss_co = loss_contr(features)  # leave this on gpu for back prop
                    batch_stats['loss_contr'] = loss_co.detach().cpu().item()  # store copy of loss value on cpu
                elif 'supervised_contrastive' in T.learning:
                    loss_co = loss_contr(features, targets)  # leave this on gpu for back prop
                    batch_stats['loss_contr'] = loss_co.detach().cpu().item()  # store copy of loss value on cpu
                stats['loss_contr'] = ((stats['loss_contr'] * batch) + batch_stats['loss_contr']) / (batch + 1)

            # display performance metrics for this batch
            postfix_string = ''
            for metric in metrics:
                postfix_string += f" | {metric}={batch_stats[metric]:.4f}({stats[metric]:.4f})"
            tepoch.set_postfix_str(postfix_string)

            # save some input images with class estimates
            if batch == 0 and hasattr(D, 'save_input_samples') and D.save_input_samples:
                if 'classification' in T.learning:
                    from utils import response
                    if M.out_channels > 1:
                        outputs = outputs[:, :, 0]
                    responses = response(outputs, D.dataset)
                else:
                    responses = None
                from utils import save_image_custom
                sample_input_dir = f'{M.model_dir}/sample_test_inputs'
                if op.isdir(sample_input_dir):
                    shutil.rmtree(sample_input_dir)
                os.makedirs(sample_input_dir)
                save_image_custom(inputs.detach().cpu(), T, sample_input_dir, max_images=128, labels=responses)

    return stats


if __name__ == '__main__':

    """
    configure a model for testing
    """
    base = f'/home/tonglab/david/projects/p022_occlusion'
    os.chdir(base)
    sys.path.append(base)

    # often used occluders and visibility levels
    occluders_fMRI = ['barHorz08']
    occluders_behavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot', 'polkasquare',
                             'crossBarOblique', 'crossBarCardinal', 'naturalUntexturedCropped2']
    visibilities = [.1, .2, .4, .6, .8]



    # model
    M = SimpleNamespace(
        model_dir = 'in_silico/data/cornet_s_custom/occ-beh',
        params_path = '/home/tonglab/david/projects/p022_occlusion/in_silico/data/cornet_s_custom/occ-beh/params/032.pt',
    )

    # dataset
    D = SimpleNamespace(
        dataset = 'ILSVRC2012',
        #contrast = 'occluder_translate',  # contrastive learning manipulation. Options: 'repeat_transform','occluder_translate'
        transform = 'alter',
        #class_subset = []
    )
    """
    class Occlusion:
        type = occluders_behavioural                       # occluder type or list thereof
        prop_occluded = .8                                 # proportion of images to be occluded
        visibility = visibilities                          # image visibility or list thereof, range(0,1)
        colour = [(0,0,0),(127,127,127),(255,255,255)]      # occluder colours (unless naturalTextured type)

    class Blur:
        sigmas = [0,1,2,4,8]
        weights = [1,1,1,1,1]

    class Noise:
        type = 'gaussian'
        ssnr = [.1,.2,.4,.8,1]
        label = 'mixedNoise'
    """

    # training
    T = SimpleNamespace(
        batch_size = 256,  # calculated if not set
        learning = 'supervised_classification',  # supervised_classification, supervised_contrastive, self-supervised_contrastive
        nGPUs = -1,  # if not set, or set to -1, all GPUs visible to pytorch will be used. Set to 0 to use CPU
        GPUids = 1,  # list of GPUs to use (ignored if nGPUs in [-1,0] or not set)
        workers = 2,  # number of CPU threads to use (calculated if not set)
    )

    os.chdir(f'/home/tonglab/david/projects/p022_occlusion')
    CFG = SimpleNamespace(M=M,D=D,T=T)
    performance = test_model(CFG)
