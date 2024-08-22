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
from datetime import datetime
dtnow, nowstr = datetime.now, "%y/%m/%d %H:%M:%S"
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

import accuracy
import get_outputs
import AverageMeter

def evaluate_model(CFG, model=None, verbose=False):  # model params, dataset
    # params, training params

    torch.no_grad()

    # unpack config
    M, D, T = CFG.M, CFG.D, CFG.T

    # get model architecture from model dir
    if op.isfile(f'{M.model_dir}/config.pkl'):
        CFG_orig = pkl.load(open(f'{M.model_dir}/config.pkl', 'rb'))
    else:
        CFG_orig = CFG

    # configure hardware
    num_GPUs = torch.cuda.device_count()
    device = torch.device('cuda') if num_GPUs else torch.device('cpu')
    num_workers = num_GPUs * 8 if os.cpu_count() <= 32 else num_GPUs * 16
    print(f'{num_workers} workers') if verbose else print()

    # model
    if model is None:
        from utils import get_model
        model = get_model(CFG_orig.M.model_name, **{'M': CFG_orig.M})
    print(model) if verbose else print()
    model.eval()

    # calculate optimal batch size
    if not hasattr(T, 'batch_size'):
        from utils import calculate_batch_size
        T = calculate_batch_size(model, T, device)
        print(f'optimal batch size calculated at {T.batch_size}')

    # loader
    from utils import get_loaders
    _, loader = get_loaders(D, T, num_workers)

    # loss functions (send to device before constructing optimizer)
    from utils import get_criteria
    criteria = get_criteria(T, device)

    ### performance metrics ###

    # performance metrics
    metrics = []
    if T.classification:
        metrics += ['acc1', 'acc5', 'loss_class']
    if T.contrastive:
        metrics += ['loss_contr']

    # load model parameters
    if not hasattr(CFG.M, 'params_loaded') or CFG.M.params_loaded is False:
        print('loading parameters...')
        if hasattr(M, 'params_path'):
            params_path = M.params_path
        else:
            params_path = sorted(glob.glob(f'{M.model_dir}/params/*.pt*'))[-1]
        from utils import load_params
        model = load_params(params_path, model, 'model')

    # put model on device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)


    ### TEST ###

    # initialize cumulative performance stats for this epoch
    epoch_tracker = {metric: AverageMeter() for metric in metrics}

    # loop through batches
    with tqdm(loader, unit=f"batch({T.batch_size})") as tepoch:

        for batch, (inputs, targets) in enumerate(tepoch):

            now = dtnow().strftime(nowstr)
            tepoch.set_description(f'{now} | evaluating ')

            # concatenate multiple views along image dimension
            if D.num_views > 1:
                inputs = torch.concat(inputs, dim=0).squeeze()

            # put inputs on device
            inputs = inputs.to(device, non_blocking=T.contrastive)
            targets = targets.to(device, non_blocking=T.contrastive)

            # automatic mixed precision
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=T.AMP):

                # pass through model
                outputs = model(inputs)

                # assign outputs according to task
                outputs_class, outputs_contr = assign_outputs(outputs, CFG)

                # contrastive accuracy and loss
                if T.contrastive:
                    targets_contr = targets if \
                        T.contrastive_supervised else None
                    loss_contr = criteria['contr'](outputs_contr,
                                                   targets_contr)
                    epoch_tracker['loss_contr'].update(
                        loss_contr.detach().cpu().item())

                # classification accuracy and loss
                if T.classification:

                    # accuracy
                    trg = torch.concat([targets] * D.num_views, dim=0)
                    acc1, acc5 = [x.detach().cpu() for x in accuracy(
                        outputs_class, trg, (1, 5))]
                    epoch_tracker['acc1'].update(acc1)
                    epoch_tracker['acc5'].update(acc5)

                    # loss
                    loss_class = criteria['class'](outputs_class, targets)
                    if T.cutmix and apply_cutmix and not T.cutmix_frgrnd:
                        trg = torch.concat([trg_frgrnd] * D.num_views,
                                           dim=0)
                        loss_class += criteria['class'](outputs_class,
                                                        trg) * lam
                    epoch_tracker[f'loss_class'].update(
                        loss_class.detach().cpu().item())

            # display performance metrics for this batch
            postfix_string = ''
            for metric in [m for m in metrics if m != 'acc5']:
                postfix_string += (
                    f"{metric}={epoch_tracker[metric].val:.4f}"
                    f"({epoch_tracker[metric].avg:.4f}) ")
            tepoch.set_postfix_str(postfix_string)

    return epoch_tracker


if __name__ == '__main__':

    """
    configure a model for testing
    """
    base = f'/home/tonglab/david/projects/p022_occlusion'
    os.chdir(base)
    sys.path.append(base)

    # model
    M = SimpleNamespace(
        model_name = 'cornet_s',
        model_dir = 'in_silico/models/cornet_s/pretrained',
        params_path = '/home/tonglab/david/projects/p022_occlusion/in_silico'
                      '/models/cornet_s/pretrained/params/cornet_s-1d3f7974'
                      '.pth',
    )

    # dataset
    D = SimpleNamespace(
        dataset = 'ILSVRC2012',
        #contrast = 'occluder_translate',  # contrastive learning manipulation. Options: 'repeat_transform','occluder_translate'
        #transform = 'alter',
        #class_subset = []
    )

    # training
    T = SimpleNamespace(
        batch_size = 256,  # calculated if not set
        learning = 'supervised_classification',  # supervised_classification, supervised_contrastive, self-supervised_contrastive
        nGPUs = -1,  # if not set, or set to -1, all GPUs visible to pytorch will be used. Set to 0 to use CPU
        GPUids = 1,  # list of GPUs to use (ignored if nGPUs in [-1,0] or not set)
    )

    os.chdir(f'/home/tonglab/david/projects/p022_occlusion')
    CFG = SimpleNamespace(M=M,D=D,T=T)
    from utils import complete_config
    CFG, model = complete_config(CFG, training=False)
    performance = test_model(CFG)
    df = pd.DataFrame()
    for k, v in performance.items():
        print(f'{k}: {v.avg}')
        df[k] = [v.avg]
    df.to_csv(f'{M.model_dir}/performance.csv')
    print('Done.')
