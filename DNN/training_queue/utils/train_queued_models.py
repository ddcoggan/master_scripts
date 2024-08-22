"""
train models in queue
"""
import os
import os.path as op
import glob
import socket
import sys
import shutil
import itertools

# set cuda GPU visibility
GPUs = input(f'Which GPU(s) has this training job been assigned to?\n'
             f'E.g., 0 or 0,1 :')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # converts to nvidia-smi order
os.environ['CUDA_VISIBLE_DEVICES'] = GPUs

# configure torch settings AFTER setting CUDA_VISIBLE_DEVICES
import torch
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(False)
# force cuda initialization
# torch.nn.functional.conv2d(
#    torch.zeros(32, 32, 32, 32, device=torch.device('cuda')),
#    torch.zeros(32, 32, 32, 32, device=torch.device('cuda')))

# machines
machines = ['finn','rey','padme','leia','solo','luke','yoda']
machine = socket.gethostname()

# model queue
queue_dir = op.expanduser('~/david/master_scripts/DNN/training_queue')
sys.path.append(queue_dir)
os.chdir(queue_dir)


def find_configs(machine, GPUs):
    configs = sorted(glob.glob(f'{machine}-{GPUs}_*.py'))
    if not configs:
        configs = sorted(glob.glob(f'*.py'))
        for config, machine in itertools.product(configs, machines):
            if machine in config:
                configs.remove(config)
    return configs


# get model list
configs = find_configs(machine, GPUs)
while configs:

    # find next model and claim it by renaming file
    config = configs[0]
    if not config.startswith(machine):
        claimed_config = f'{machine}-{GPUs}_{config}'
        shutil.move(config, claimed_config)
        config = claimed_config

    # load config, get model directory
    CFG = __import__(config[:-3], fromlist=['']).CFG
    if CFG.M.model_dir is None:
        CFG.M.model_dir = op.expanduser(
            f'~/david/models/{CFG.M.architecture}/{CFG.M.identifier}')
    if CFG.M.finetune and not CFG.M.model_dir.endswith(CFG.M.finetune_dir):
        CFG.M.model_dir += f'/{CFG.M.finetune_dir}'

    # make a copy of training utils in model directory for reproducibility
    os.makedirs(CFG.M.model_dir, exist_ok=True)
    utils_dir = f'{CFG.M.model_dir}/utils'
    if not op.exists(utils_dir):
        shutil.copytree(
            op.expanduser('~/david/master_scripts/DNN/utils'), utils_dir)
    sys.path.append(utils_dir)

    # calculate / set missing values in config
    from complete_config import complete_config
    CFG = complete_config(CFG, resolve='new')

    # train model
    from optimize_model import optimize_model
    optimize_model(CFG, verbose=True)

    # clean up after training
    shutil.copy(config, f'{CFG.M.model_dir}/config.py')
    shutil.move(config, f'done/{config}')
    del CFG, complete_config, optimize_model

    # refresh model configs
    configs = find_configs(machine, GPUs)


