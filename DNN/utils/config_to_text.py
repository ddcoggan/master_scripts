# /usr/bin/python
# Created by David Coggan on 2023 02 09
# saves viewable text version of model configuration (appends to file in cases where training stopped and resumed)
# accepts paths config or the config object itself as input
import pickle as pkl
import datetime
import os
import os.path as op
from types import SimpleNamespace

def config_to_text(CFG):

    # if CFG is path, then load
    if isinstance(CFG, str):
        CFG = pkl.load(open(CFG, 'rb'))

    os.makedirs(CFG.M.model_dir, exist_ok=True)
    config_txt = f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n\n'
    for param_type, param_space in zip(['model', 'dataset', 'training'],
                                       [CFG.M, CFG.D, CFG.O]):
        config_txt += f'### {param_type} ###\n'
        for param_name, param in param_space.__dict__.items():
            if not param_name.startswith('_') and param_name not in ['model']:
                if type(param) is SimpleNamespace:
                    param_type_printed = False
                    for subparam_name, subparam in param.__dict__.items():
                        if not subparam_name.startswith('_'):
                            if not param_type_printed:
                                config_txt += f'{param_name.ljust(32)}{subparam_name.ljust(32)}{subparam}\n'
                                param_type_printed = True
                            else:
                                config_txt += f'{subparam_name.ljust(32).rjust(64)}{subparam}\n'  # if param is another class
                else:
                    config_txt += f'{param_name.ljust(32)}{param}\n'  # if param is a parameter
        config_txt += '\n\n'
    config_txt += '\n\n\n\n'
    config_path_txt = op.join(CFG.M.model_dir, 'config.txt')
    with open(config_path_txt, 'a') as c:
        c.write(config_txt)

if __name__ == '__main__':

    config = '/home/tonglab/david/models/cornet_s_custom'
    #config_to_text(config)
