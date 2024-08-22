

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



def complete_config(CFG, resolve='new', optimize=True):

    """
    This function determines basic optimization parameters prior to calling the
    optimization script. Specifically, it calculates the maximum batch size
    and reasonable learning rate, and resolves conflicts when an optimization
    regime is resumed with new settings.

    resolve = 'resume' to continue optimizing with no changes to the config
    resolve = 'new' to merge configs, with conflicts resolved in favour of new
    resolve = 'orig' to merge configs, with conflicts resolved in favour of orig

    unless resolve == 'resume', the following parameters are ALWAYS
    overwritten by values in the NEW config:
        ['checkpoint', 'num_epochs', 'batch_size', 'num_workers']

    if CFG.O.overwrite_optimizer is True, the following parameters are ALSO
    overwritten by values in the NEW config:
        ['optimizer', 'optimizer_args', 'scheduler', 'scheduler_args', 'swa', 'amp']
    """

    # look for previous configs
    config_file = f"{CFG.M.model_dir}/config.pkl"
    orig_config_exists = op.isfile(config_file)
    if orig_config_exists:
        CFG_orig = pkl.load(open(config_file, 'rb'))

    # unless starting new optimization regime or resuming with no changes, merge configs
    if not orig_config_exists:
        if CFG.M.finetune:
            if CFG.M.starting_params is None:
                try:
                    CFG.M.starting_params = sorted(glob.glob(
                        f'{op.dirname(CFG.M.model_dir)}/params/*.pt'))[-1]
                    print(f'Transfer learning regime initiated using params at '
                          f'{CFG.M.starting_params}')
                except: 
                    ValueError, ('finetuning regime requested but starting '
                                 'weights were not found')
        else:
            print(f'Starting optimization of new model at {CFG.M.model_dir}')

    elif resolve == 'resume':
        print(f'Resuming optimization of model at {CFG.M.model_dir} with no '
              f'config changes')
        assert orig_config_exists, 'Cannot find original config to resume'
        CFG = CFG_orig
    else:
        print(f'Resuming optimization of model at {CFG.M.model_dir} with '
              f'potential config changes.\nConflicts between original and new '
              f'configs resolved in favour of {resolve}.')

        # force overwrite for certain parameters
        forced_params = [
            'checkpoint', 'num_epochs', 'batch_size', 'num_workers']
        if CFG.O.overwrite_optimizer:
            forced_params += ['optimizer', 'optimizer_args', 'scheduler',
                              'scheduler_args', 'swa', 'amp']
        print(f'The following parameters will be forced if they exist in the '
              f'new config: {forced_params}')

        # resolve other param conflict (orig params written into new config)
        for params in ['M','D','O']:
            params_orig = getattr(CFG_orig, params)
            params_new = getattr(CFG, params)
            all_params = [*set(list(params_orig.__dict__.keys()) + list(
                params_new.__dict__.keys()))]
            for param in all_params:
                if param not in forced_params and not param.startswith('_'):
                    if hasattr(params_orig, param) and (
                            not hasattr(params_new, param)):
                        print(f'Parameter ({param}) found only in orig config, '
                              f'using in final config')
                        item = getattr(params_orig, param)
                        setattr(params_new, param, item)

                    elif not hasattr(params_orig, param) and (
                            hasattr(params_new, param)):
                        print(f'Parameter ({param}) found only in new config, '
                              f'using in final config')

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
    M, D, O = CFG.M, CFG.D, CFG.O

    # add any missing default parameters
    defaults = SimpleNamespace(
        D=SimpleNamespace(
            dataset='ILSVRC2012',
            num_views=1,
            transform_type='default',
            image_size=224,
            cutmix=None,
        ),
        O=SimpleNamespace(
            num_epochs=100,
            criterion=dict(CrossEntropyLoss=dict(views=[0],weight=1)),
            optimizer='SGD',
            optimizer_args=dict(
                lr=.05,
                momentum=0.9,
                weight_decay=1e-4),
            scheduler='StepLR',
            scheduler_args=dict(step_size=30, gamma=0.1,),
            swa=None,
            amp=True,
        ),
    )
    for default_params, params in zip([defaults.D, defaults.O], [D, O]):
        for param in default_params.__dict__.keys():
            if not hasattr(params, param):
                setattr(params, param, getattr(default_params, param))

    # calculate last epoch if not set or resume optimization with no changes
    if not hasattr(O, 'checkpoint') or O.checkpoint is None or (
            resolve == 'resume'):
        params_paths = sorted(glob.glob(f"{CFG.M.model_dir}/params/???.pt"))
        if params_paths:
            print(f'Most recent params found at {params_paths[-1]}')
            O.checkpoint = int(os.path.basename(params_paths[-1])[:-3])
        else:
            O.checkpoint = -1

    # determine whether model has finished optimizing
    epoch_stats_path = f'{CFG.M.model_dir}/epoch_stats.csv'
    if op.isfile(epoch_stats_path):
        epoch_stats = pd.read_csv(open(epoch_stats_path, 'r+'))
        optimize = len(epoch_stats) < (O.num_epochs+1)*2
    else:
        optimize = True

    if optimize:

        import torch
        sys.path.append(op.expanduser('~/david/master_scripts/DNN'))

        # remove warnings for predified networks
        if M.architecture.endswith('predify'):
            import warnings
            warnings.simplefilter("ignore")

        # loading model necessary for hyperparameter calculation
        def load_model(M, D, O):

            # configure hardware
            device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')
            
            # load model
            from utils import get_model
            model = get_model(M.architecture)

            # adapt output size of model based on number of classes in dataset
            num_classes = len(glob.glob(f'{op.expanduser("~")}/Datasets/{D.dataset}/train/*'))
            if num_classes != 1000 and O.classification:
                from utils import change_output_size
                model = change_output_size(model, M, num_classes)
            
            return model, device

        # calculate optimal batch size
        if not hasattr(O, 'batch_size'):
            from utils import calculate_batch_size
            model, device = load_model(M, D, O)
            O = calculate_batch_size(model, O, O.device)
            print(f'optimal batch size calculated at {O.batch_size}')

        # calculate optimal learning rate
        if optimize and type(O.optimizer_args['lr']) == str:
            model, device = load_model(M, D, O)
            if O.optimizer_args['lr'] == 'batch_nonlinear':
                O.optimizer_args['lr'] = 2**-7 * np.sqrt(O.batch_size/2**5)
            elif O.optimizer_args['lr'] == 'batch_linear':
                O.optimizer_args['lr'] = O.batch_size / 2**5
            elif O.optimizer_args['lr'] == 'LRfinder':
                from utils import find_lr
                O.optimizer_args['lr'] = find_lr(model, device, D, O)

    # repack config
    CFG = SimpleNamespace(M=M, D=D, O=O)

    return CFG

