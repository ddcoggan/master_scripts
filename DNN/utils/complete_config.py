"""
This script determines basic training parameters prior to calling the training script.
Specifically, it calculates the maximum batch size and reasonable learning rate, and
resolves conflicts when a training regime is resumed with new settings
"""

import os
import os.path as op
import glob
import sys
import datetime
import numpy as np
import pickle as pkl
import shutil
from types import SimpleNamespace
import pandas as pd

# resolve = 'resume' to continue training with no changes to the configuration
# resolve = 'new' to merge newly specified and original configs, with conflicts resolved in favour of new
# resolve = 'orig' to merge newly specified and original configs, with conflicts resolved in favour of orig
# unless resolve == 'resume', ['checkpoint', 'num_epochs', 'batch_size'] are ALWAYS overwritten by values in new config

def complete_config(CFG, resolve='new', training=True):

    # look for previous configs
    config_file = f"{CFG.M.model_dir}/config.pkl"
    orig_config_exists = op.isfile(config_file)  # default
    if orig_config_exists:
        CFG_orig = pkl.load(open(config_file, 'rb'))

    # unless starting new training regime or resuming with no changes, merge configs
    if not orig_config_exists:
        if hasattr(CFG.M, 'transfer') and CFG.M.transfer:
            if not hasattr(CFG.M, 'starting_params'):
                CFG.M.starting_params = sorted(glob.glob(f'{op.dirname(CFG.M.model_dir)}/params/*.pt'))[-1]
            print(f'Transfer learning regime initiated using params at {CFG.M.starting_params}')
        else:
            print(f'Starting training of new model at {CFG.M.model_dir}')

    elif resolve == 'resume':
        print(f'Resuming training of model at {CFG.M.model_dir} with no config changes')
        assert orig_config_exists
        CFG = CFG_orig
    else:
        print(f'Resuming training of model at {CFG.M.model_dir} with potential config changes.\n'
              f'Conflicts between original and new configs resolved in favour of {resolve}.')

        # force overwrite for certain parameters
        forced_params = ['checkpoint', 'num_epochs', 'batch_size', 'num_workers']
        if hasattr(CFG.T, 'overwrite_optimizer') and CFG.T.overwrite_optimizer:
            forced_params += ['learning_rate','step_size','momentum','weight_decay','patience']
        print(f'The following parameters will be forced if they exist in the new config: {forced_params}')

        # resolve conflict for other params (original params are written into the new config)
        for params in ['M','D','T']:
            params_orig = getattr(CFG_orig, params)
            params_new = getattr(CFG, params)
            all_params = [*set(list(params_orig.__dict__.keys()) + list(params_new.__dict__.keys()))]
            for param in all_params:
                if param not in forced_params and not param.startswith('_'):

                    if hasattr(params_orig, param) and not hasattr(params_new, param):
                        print(f'Parameter ({param}) found only in orig config, using in final config')
                        item = getattr(params_orig, param)
                        setattr(params_new, param, item)

                    elif not hasattr(params_orig, param) and hasattr(params_new, param):
                        print(f'Parameter ({param}) found only in new config, using in final config')

                    elif hasattr(params_orig, param) and hasattr(params_new, param):
                        orig_item = getattr(params_orig, param)
                        new_item = getattr(params_new, param)
                        if orig_item == new_item:
                            print(f'No conflict for parameter ({param})')
                        elif resolve == 'orig':
                            print(f'Conflict for parameter ({param}), using orig')
                            setattr(params_new, param, orig_item)
                        else:
                            print(f'Conflict for parameter ({param}), using new')


    # unpack resolved config
    M, D, T = CFG.M, CFG.D, CFG.T

    # add any missing default parameters
    defaults = SimpleNamespace(
        D = SimpleNamespace(
            dataset='ILSVRC2012',
            num_views = 1,
            transform_type = 'default',
            image_size = 224
        ),
        T = SimpleNamespace(
            num_epochs = 100,
            classification=True,
            contrastive = False,
            contrastive_supervised = False,
            optimizer_name = 'SGD',
            momentum = 0.9,
            gamma=0.1,
            weight_decay = 1e-4,
            scheduler = 'StepLR',
            step_size = 30,
            SWA = False,
            cutmix = False,
            AMP = True,
        ),
    )
    for default_params, params in zip([defaults.D, defaults.T], [D,T]):
        for param in default_params.__dict__.keys():
            if not hasattr(params, param):
                setattr(params, param, getattr(default_params, param))

    # calculate last epoch if not set or resuming training with no changes
    if not hasattr(T, 'checkpoint') or T.checkpoint in [None, -1] or resolve == 'resume':
        params_paths = sorted(glob.glob(f"{CFG.M.model_dir}/params/???.pt"))
        if params_paths:
            print(f'Most recent params found at {params_paths[-1]}')
            T.checkpoint = int(os.path.basename(params_paths[-1])[:-3])
        else:
            T.checkpoint = -1

    # determine whether model has finished training
    epoch_stats_path = f'{CFG.M.model_dir}/epoch_stats.csv'
    if op.isfile(epoch_stats_path):
        epoch_stats = pd.read_csv(open(epoch_stats_path, 'r+'))
        do_training = len(epoch_stats) < (T.num_epochs+1)*2
    else:
        do_training = True

    if do_training:

        import torch
        sys.path.append(op.expanduser('~/david/master_scripts/DNN'))

        # remove warnings for predified networks
        if M.model_name.endswith('predify'):
            import warnings
            warnings.simplefilter("ignore")

        # loading model necessary for hyperparameter calculation
        def load_model(M, D, T):

            # configure hardware
            device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')
            
            # load model
            from utils import get_model
            model = get_model(M.model_name)

            # adapt output size of model based on number of classes in dataset
            num_classes = len(glob.glob(f'{op.expanduser("~")}/Datasets/{D.dataset}/train/*'))
            if num_classes != 1000 and T.classification:
                from utils import change_output_size
                model = change_output_size(model, M, num_classes)
            
            return model, device

        # calculate optimal batch size
        if not hasattr(T, 'batch_size'):
            from utils import calculate_batch_size
            model, device = load_model(M, D, T)
            T = calculate_batch_size(model, T, T.device)
            print(f'optimal batch size calculated at {T.batch_size}')

        # calculate optimal learning rate
        if training and type(T.learning_rate) == str:
            model, device = load_model(M, D, T)
            if T.learning_rate == 'batch_nonlinear':
                T.learning_rate = 2**-7 * np.sqrt(T.batch_size/2**5)
            elif T.learning_rate == 'batch_linear':
                T.learning_rate = T.batch_size / 2**5
            elif T.learning_rate == 'LRfinder':
                from utils import find_lr
                T.learning_rate = find_lr(model, device, D, T)

    # repack config
    CFG = SimpleNamespace(M=M, D=D, T=T)

    return CFG

