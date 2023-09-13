import torch
import numpy as np
import os
from argparse import Namespace

def configure_hardware(T=Namespace(), verbose=True):

    # GPUs
    totalGPUs = torch.cuda.device_count()
    if verbose: print(f'{totalGPUs} GPUs available to pytorch')
    if not totalGPUs or (hasattr(T, 'nGPUs') and T.nGPUs == 0):
        if verbose: print(f'Using CPU')
        device = torch.device('cpu')
        T.nGPUs = 0
    elif not hasattr(T, 'nGPUs') or T.nGPUs == -1:
        if verbose: print(f'Using all available GPUs')
        device = torch.device('cuda:0')
        T.nGPUs = totalGPUs
        T.GPUids = list(np.arange(totalGPUs))
    elif T.nGPUs == 1:
        if not hasattr(T, 'GPUids'):
            T.GPUids = 0
        if verbose: print(f'Using GPU #{T.GPUids}')
        device = torch.device(f'cuda:{T.GPUids}')
    elif T.nGPUs > 1:
        if verbose: print(f'Using {T.nGPUs} GPUs #{T.GPUids}')
        device = torch.device(f'cuda:{T.GPUids[0]}') # set primary device

    return T, device

