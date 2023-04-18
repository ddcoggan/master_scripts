import os
import os.path as op
import glob
import sys
import datetime
import numpy as np
import pickle as pkl
import shutil
sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
# as an alternative to controlling GPU utilization with pytorch, set pytorch to use any available GPUs and
# use these lines to control which GPUs are visible to pytorch (run before importing pytorch)
# This is most useful if calling many different objects that each need to know the GPU configuration,
# as it does not require the passing of GPU configurations between them
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # converts to order in nvidia-smi (not in cuda)
#os.environ["CUDA_VISIBLE_DEVICES"] = f"[0,1]" # which devices to use

# resolve = 'resume' to continue training with no changes to the configuration
# resolve = 'new' to merge newly specified and original configs, with conflicts resolved in favour of new
# resolve = 'orig' to merge newly specified and original configs, with conflicts resolved in favour of orig

def complete_config(CFG, model_dir=None, resolve='new'):

    if model_dir is None:
        model_dir = op.expanduser(f'~/david/projects/p022_occlusion/in_silico/data/{CFG.M.model_name}/{CFG.M.identifier}')
    CFG.M.model_dir = model_dir

    # look for previous configs
    config_file = f"{model_dir}/config.pkl"
    orig_config_exists = op.isfile(config_file)  # default
    if orig_config_exists:
        CFG_orig = pkl.load(open(config_file, 'rb'))

    # unless starting new training regime or resuming with no changes, merge configs
    if not orig_config_exists:
        print(f'Starting training of new model at {model_dir}')
    elif resolve == 'resume':
        print(f'Resuming training of model at {model_dir} with no config changes')
        assert orig_config_exists
        CFG = CFG_orig
    else:
        print(f'Resuming training of model at {model_dir} with potential config changes')

        # force overwrite for certain parameters
        forced_params = ['last_epoch', 'num_epochs', 'batch_size']
        if CFG.T.force_lr:
            forced_params.append('learning_rate')

        for params in ['M','D','T']:
            params_orig = getattr(CFG_orig, params)
            params_new = getattr(CFG, params)
            all_params = [*set(list(params_orig.__dict__.keys()) + list(params_new.__dict__.keys()))]
            for param in all_params:
                if param not in forced_params and not param.startswith('_'):

                    if hasattr(params_orig, param) and not hasattr(params_new, param):
                        print(f'Parameter ({param}) found only in orig config, using in final config')
                        item = getattr(params_orig, param)

                    elif not hasattr(params_orig, param) and hasattr(params_new, param):
                        print(f'Parameter ({param}) found only in new config, using in final config')
                        item = getattr(params_new, param)

                    elif hasattr(params_orig, param) and hasattr(params_new, param):
                        orig_item = getattr(params_orig, param)
                        new_item = getattr(params_new, param)
                        if orig_item == new_item:
                            print(f'No conflict for parameter ({param})')
                            item = orig_item
                        else:
                            if resolve == 'orig':
                                print(f'Conflict for parameter ({param}), using orig')
                                item = orig_item
                            else:
                                print(f'Conflict for parameter ({param}), using new')
                                item = new_item
                    setattr(params_new, param, item)


    # calculate last epoch if not set or resuming training with no changes
    if not hasattr(CFG.T, 'last_epoch') or resolve == 'resume':
        params_paths = sorted(glob.glob(f"{CFG.M.model_dir}/params/*.pt"))
        if params_paths:
            CFG.T.last_epoch = int(os.path.basename(params_paths[-1])[:-3])
        elif CFG.T.freeze_weights: # special case for transfer learning
            print('Copying weights over for transfer learning')
            params_dir_new = f'{CFG.M.model_dir}/params'
            os.makedirs(params_dir_new)
            shutil.copy(params_path, f'{params_dir_new}/000.pt')
            CFG.T.last_epoch = 0
        else:
            CFG.T.last_epoch = None

    # determine whether model has finished training
    epoch_stats_path = f'{model_dir}/epoch_stats.pkl'
    if op.isfile(epoch_stats_path):
        epoch_stats = pkl.load(open(epoch_stats_path, 'rb'))
        key = [key for key in epoch_stats['eval'] if key.startswith('loss')][0]
        do_training = 0 not in epoch_stats['eval'][key] or CFG.T.last_epoch is None or CFG.T.num_epochs < CFG.T.last_epoch
    else:
        do_training = True

    if do_training:

        sys.path.append(op.expanduser('~/david/masterScripts/DNN'))

        # remove warnings for predified networks
        if CFG.M.model_name.endswith('predify'):
            import warnings
            warnings.simplefilter("ignore")

        # configure hardware
        from utils import configure_hardware
        CFG.T, CFG.T.device = configure_hardware(CFG.T)

        # load model
        if not hasattr(CFG.M, 'model'):
            from utils import get_model
            CFG.M.model = get_model(CFG.M)

        # adapt output size of model based on number of classes in dataset
        num_classes = len(glob.glob(f'{op.expanduser("~")}/Datasets/{CFG.D.dataset}/train/*'))
        if num_classes != 1000 and CFG.T.learning == 'supervised_classification':
            from utils import change_output_size
            CFG.M.model = change_output_size(CFG.M.model, CFG.M, num_classes)

        # calculate optimal batch size
        if not hasattr(CFG.T, 'batch_size'):
            from utils import calculate_batch_size
            t = calculate_batch_size(CFG.M.model, CFG.T, CFG.T.device)
            print(f'optimal batch size calculated at {t.batch_size}')

        # calculate optimal learning rate
        if not hasattr(CFG.T, 'learning_rate'):
            CFG.T.learning_rate = 2**-7 * np.sqrt(CFG.T.batch_size/2**5) # non-linear
            #CFG.T.learning_rate * CFG.T.batch_size / 2 ** 5  # linear
            print(f'initial learning rate calculated at {CFG.T.learning_rate}')

        # save viewable text version of final configuration (appends to file in cases where training stopped and resumed)
        os.makedirs(CFG.M.model_dir, exist_ok=True)
        config_txt = f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n\n'
        for param_type, param_space in zip(['model', 'dataset', 'training'], [CFG.M, CFG.D, CFG.T]):
            config_txt += f'### {param_type} ###\n'
            for param_name, param in param_space.__dict__.items():
                if not param_name.startswith('_') and param_name != 'model':
                    if type(param) is type:
                        param_type_printed = False
                        for subparam_name, subparam in param.__dict__.items():
                            if not subparam_name.startswith('_'):
                                if not param_type_printed:
                                    config_txt += f'{param_name.ljust(16)}{subparam_name.ljust(16)}{subparam}\n'
                                else:
                                    config_txt += f'{subparam_name.ljust(16).rjust(32)}{subparam}\n'  # if param is another class
                    else:
                        config_txt += f'{param_name.ljust(16)}{param}\n'  # if param is a parameter
            config_txt += '\n\n'
        config_txt += '\n\n\n\n'
        config_path_txt = op.join(CFG.M.model_dir, 'config.txt')
        with open(config_path_txt, 'a') as c:
            c.write(config_txt)

    return CFG
