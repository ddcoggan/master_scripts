import numpy as np
import os
from types import SimpleNamespace

def configure_hardware(T=SimpleNamespace(), verbose=True):

    # GPUs
    total_GPUs = int(os.popen('nvidia-smi -L | wc -l').read().split()[0])
    if verbose: print(f'{total_GPUs} GPUs found')
    if not total_GPUs or (hasattr(T, 'nGPUs') and T.nGPUs == 0):
        if verbose: print(f'Using CPU')
        T.nGPUs = 0
    elif not hasattr(T, 'nGPUs') or T.nGPUs == -1:
        if verbose: print(f'Using all available GPUs')
        T.nGPUs = total_GPUs
        T.GPUids = list(np.arange(total_GPUs))
    elif T.nGPUs == 1:
        if not hasattr(T, 'GPUids'):
            T.GPUids = [0]
        if verbose: print(f'Using GPU #{T.GPUids[0]}')
    elif T.nGPUs > 1:
        if verbose: print(f'Using {T.nGPUs} GPUs #{T.GPUids}')

    return T

