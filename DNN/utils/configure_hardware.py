import torch
import numpy as np
import os
from argparse import Namespace

def configure_hardware(t=Namespace()):

    # GPUs
    totalGPUs = torch.cuda.device_count()
    print(f'{totalGPUs} GPUs available to pytorch')
    if not totalGPUs or (hasattr(t, 'nGPUs') and t.nGPUs == 0):
        print(f'Using CPU')
        device = torch.device('cpu')
        t.nGPUs = 0
    elif not hasattr(t, 'nGPUs') or t.nGPUs == -1:
        print(f'Using all available GPUs')
        device = torch.device('cuda:0')
        t.nGPUs = totalGPUs
        t.GPUids = list(np.arange(totalGPUs))
    elif t.nGPUs == 1:
        if not hasattr(t, 'GPUids'):
            t.GPUids = 0
        print(f'Using GPU #{t.GPUids}')
        device = torch.device(f'cuda:{t.GPUids}')
    elif t.nGPUs > 1:
        print(f'Using {t.nGPUs} GPUs #{t.GPUids}')
        device = torch.device(f'cuda:{t.GPUids[0]}') # set primary device

    # CPU cores (workers)
    # workers : total cores = nGPUs : total GPUs
    if not hasattr(t, 'workers'):
        ncores = os.cpu_count() - 2  # (leaving some cores free)
        t.workers = int(ncores * (t.nGPUs / totalGPUs))
    print(f'{t.workers} workers')

    return t, device

