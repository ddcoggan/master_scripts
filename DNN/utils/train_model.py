import sys
import pickle as pkl
import pandas as pd
import numpy as np
import glob
import os
import os.path as op
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import Namespace
import math
import pprint
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributed as dist
from torch.autograd.variable import Variable
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy


sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
from utils import accuracy
from utils.assign_outputs import assign_outputs
from utils.AverageMeter import AverageMeter
from utils.cutmix import cutmix
from utils.plot_performance import plot_performance
from utils.save_image_custom import save_image_custom

torch.backends.cudnn.benchmark = True
#torch.autograd.set_detect_anomaly(True)
torch.random.manual_seed(42)


def train_model(CFG, model=None, verbose=False):

    # unpack config
    M, D, T = CFG.M, CFG.D, CFG.T

    # initialise model directory and save config
    os.makedirs(f'{M.model_dir}/params', exist_ok=True)
    pkl.dump(CFG, open(f'{CFG.M.model_dir}/config.pkl', 'wb'))
    from utils import config_to_text
    config_to_text(CFG)

    # configure GPUs
    if hasattr(T, 'device'):
        device = T.device
    else:
        from utils import configure_hardware
        T, device = configure_hardware(T)

    # configure CPU cores
    num_cores = os.cpu_count() - 2  # (leave some cores free)
    num_workers = int(num_cores * (T.nGPUs / torch.cuda.device_count()))
    if verbose:
        print(f'{num_workers} workers')

    # image processing
    from utils import get_transforms
    train_path = op.expanduser(f'~/Datasets/{D.dataset}/train')
    val_path = op.expanduser(f'~/Datasets/{D.dataset}/val')
    transform_train, transform_val = get_transforms(D)
    batch_size_adjusted = T.batch_size // D.num_views
    train_data = ImageFolder(train_path, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=batch_size_adjusted,
                              shuffle=True, num_workers=num_workers)
    val_data = ImageFolder(val_path, transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=batch_size_adjusted//2,
                            shuffle=True, num_workers=num_workers)

    # loss functions (send to device before constructing optimizer)
    if T.classification:
        loss_class = nn.CrossEntropyLoss().to(device)
    if T.contrastive:
        from utils import ContrastiveLoss
        loss_contr = ContrastiveLoss().to(device)

    # performance metrics
    metrics = []
    if T.classification:
        metrics += ['acc1', 'acc5', 'loss_class']
        loss_metric, sched_metric, sched_mode = 'loss_class', 'acc1', 'max'
    if T.contrastive:
        metrics += ['loss_contr']
        loss_metric, sched_metric, sched_mode = \
            'loss_contr', 'loss_contr', 'min'

    # performance table
    performance_path = f'{M.model_dir}/performance.csv'
    if op.isfile(performance_path):
        performance = pd.read_csv(open(performance_path, 'r+'), index_col=0)
    else:
        performance = {'epoch': [], 'train_eval': [], 'lr': []}
        for metric in metrics:
            performance[metric] = []
        performance['time'] = []
        performance = pd.DataFrame(performance)


    # model
    if model is None:
        from utils import get_model
        model = get_model(M)
    if verbose:
        print(model)

    # find model params
    if T.checkpoint == -1:
        params_path = M.starting_params if \
            hasattr(M, 'starting_params') else None
        modules = M.freeze_layers if hasattr(M, 'freeze_layers') else 'all'
    else:
        params_path = f'{M.model_dir}/params/{T.checkpoint:03}.pt'
        modules = 'all'  # all params loaded if modules


    # load model state
    if params_path:

        print(f'Loading model state from {params_path}')
        params = torch.load(params_path)
        from utils import load_params
        model = load_params(params, model, 'model', modules=modules)

        # freeze weights (turn off gradients)
        if hasattr(M, 'freeze_layers'):
            for module in M.freeze_layers:
                print(f'Freezing layer: {module}')
                getattr(model, module).requires_grad_(False)
               
            # view param items
            if verbose:
                for p, param in enumerate(model.parameters()):
                    print(f'param: {p} | shape: {param.shape} | '
                          f'requires_grad: {param.requires_grad}')


    # put model on device
    if T.nGPUs > 1:
        model = nn.DataParallel(model)
    model.to(device)


    # optimizer
    if T.optimizer_name == 'SGD':
        optimizer = optim.SGD(params=model.parameters(), 
                              lr=T.learning_rate, 
                              momentum=T.momentum,
                              weight_decay=T.weight_decay)
    elif T.optimizer_name == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), 
                               lr=T.learning_rate,
                               weight_decay=T.weight_decay)
    if T.SWA:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
    
    # scheduler
    optimizer.param_groups[0]['initial_lr'] = T.learning_rate
    if T.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=T.step_size, last_epoch=T.checkpoint)
    if T.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=T.patience)
    if T.SWA:
        swa_scheduler = SWALR(optimizer, anneal_epochs=T.SWA_freq,
                              swa_lr=T.SWA_lr)
        
    # load optimizer and scheduler states
    if T.checkpoint > -1:
        print('Loading optimizer state')
        optimizer = load_params(params, optimizer, 'optimizer')
        scheduler = load_params(params, scheduler, 'scheduler')
        if T.SWA:
            swa_scheduler = load_params(params, swa_scheduler, 'swa_scheduler')

    
    # save initial model and optimizer state
    if T.checkpoint == -1:
        print('Saving initial model and optimizer state')
        params_path = f'{M.model_dir}/params/000.pt'
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()}
        if T.SWA:
            params['swa_model'] = swa_model.state_dict()
            params['swa_scheduler'] = swa_scheduler.state_dict()
        torch.save(params, params_path)


    # train / eval loop (do epoch zero last)
    for epoch in list(range(max(1, T.checkpoint+1), T.num_epochs + 1)) + [0]:
        # load epoch 0 params at the end
        if epoch == 0:
            from utils import load_params
            params_path = f'{M.model_dir}/params/000.pt'
            model = load_params(params_path, model, 'model')

        for train_eval in ['train', 'eval']:

            # train/eval specific settings
            if train_eval == 'train':
                loader = train_loader
                model.train() if epoch > 0 else model.eval()

            else:
                loader = val_loader
                model.eval()

            # initialize cumulative performance stats for this epoch
            epoch_tracker = {metric: AverageMeter() for metric in metrics}

            # loop through batches
            with tqdm(loader, unit=f"batch({T.batch_size})") as tepoch:

                for batch, (inputs, targets) in enumerate(tepoch):

                    now = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
                    tepoch.set_description(f'{now} | {train_eval} | epoch'
                                           f' {epoch}/{T.num_epochs}')

                    # apply cutmix
                    if T.cutmix:
                        apply_cutmix = np.random.rand(1) < T.cutmix_prob
                        if apply_cutmix:
                            inputs, trg_frgrnd, lam, = cutmix(inputs,
                                                                  targets, T)
                        targets = trg_frgrnd if T.cutmix_frgrnd else targets

                    # save some input images after one successful batch
                    if epoch == 1 and batch == 1:
                        save_image_custom(
                            inputs, f'{M.model_dir}/sample_training_inputs')

                    # concatenate multiple views along image dimension
                    if D.num_views > 1:
                        inputs = torch.concat(inputs, dim=0).squeeze()

                    # put inputs on device
                    inputs = inputs.to(device, non_blocking=T.contrastive)
                    targets = targets.to(device, non_blocking=T.contrastive)

                    # pass through model
                    outputs = model(inputs)

                    # decompose outputs according to task
                    outputs_class, outputs_cont = assign_outputs(outputs, CFG)

                    # contrastive accuracy and loss
                    if T.contrastive:
                        targets_contr = targets if \
                            T.contrastive_supervised else None
                        loss_co = loss_contr(outputs_contr, targets_contr)
                        epoch_tracker['loss_contr'].update(
                            loss_co.detach().cpu().item())

                    # classification accuracy and loss
                    if T.classification:    

                        # accuracy
                        trg = torch.concat([targets] * D.num_views, dim=0)
                        acc1, acc5 = [x.detach().cpu() for x in accuracy(
                            outputs_class, trg, (1, 5))]
                        epoch_tracker['acc1'].update(acc1)
                        epoch_tracker['acc5'].update(acc5)

                        # loss
                        loss_cl = loss_class(outputs_class, targets)
                        if T.cutmix and apply_cutmix and not T.cutmix_frgrnd:
                            trg = torch.concat([trg_frgrnd] * D.num_views,
                                               dim=0)
                            loss_cl += loss_class(outputs_class, trg) * lam
                        epoch_tracker[f'loss_class'].update(
                            loss_cl.detach().cpu().item())

                    # display performance metrics for this batch
                    postfix_string = ''
                    current_lr = optimizer.param_groups[0]['lr'] * \
                                 (train_eval == 'train' and epoch != 0)
                    for metric in [m for m in metrics if m != 'acc5']:
                        postfix_string += (
                            f"{metric}={epoch_tracker[metric].val:.4f}"
                            f"({epoch_tracker[metric].avg:.4f}) ")
                    postfix_string += f" lr={current_lr:.5f}"
                    tepoch.set_postfix_str(postfix_string)

                    # update model
                    if train_eval == 'train' and epoch > 0:
                        if not (T.classification and T.contrastive):
                            loss = loss_cl if T.classification else loss_co
                        else:
                            if T.loss_ratio:
                                loss_ratio_norm = F.normalize(torch.tensor(
                                    T.loss_ratio).float(), p=1, dim=0)
                            else:
                                loss_ratio_norm = (.5, .5)
                            loss = loss_cl * loss_ratio_norm[0] + loss_co * \
                                loss_ratio_norm[1]
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

            # add performance for this epoch
            new_stats = {'time': [now], 'epoch': [epoch],
                         'train_eval': [train_eval], 'lr': [current_lr]}
            new_stats = {**new_stats,
                         **{key: np.array(value.avg, dtype="float16")
                            for key, value in (epoch_tracker.items())}}
            performance = pd.concat(
                [performance, pd.DataFrame(new_stats)]).reset_index(drop=True)
            performance = performance.sort_values(
                by=['epoch','train_eval'], ascending=[True,False]).reset_index(
                drop=True)
            performance.to_csv(performance_path)



        # update LR scheduler and perform SWA
        if T.SWA and epoch > T.SWA_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if epoch == T.num_epochs:
                update_bn(train_loader, swa_model)
        else:
            if T.scheduler == 'StepLR':
                scheduler.step()
            elif T.scheduler == 'ReduceLROnPlateau':
                scheduler.step(performance[sched_metric].values[-1])

        # save new state
        params_path = f"{M.model_dir}/params/{epoch:03}.pt"
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()}
        if T.SWA:
            params['swa_model'] = swa_model.state_dict()
            params['swa_scheduler'] = swa_scheduler.state_dict()
        torch.save(params, params_path)

        # save best state
        compare = np.less if sched_mode == 'min' else np.greater
        prev = performance[sched_metric][performance['train_eval'] ==
                                         'eval'].to_numpy()
        new = epoch_tracker[sched_metric].avg
        if compare(new, prev).all():
            best_path = f"{M.model_dir}/params/best.pt"
            torch.save(params, best_path)

        # delete previous state
        if not hasattr(M, 'save_interval') or (epoch > 1 and
                M.save_interval > 1 and (epoch - 1) % M.save_interval != 0):
            last_save_path = f"{M.model_dir}/params/{epoch - 1:03}.pt"
            if op.exists(last_save_path):
                os.remove(last_save_path)

        # plot performance
        if epoch != 1:
            plot_performance(performance, metrics, M.model_dir)

    print('Training complete.')


if __name__ == '__main__':

    from types import SimpleNamespace
    M = SimpleNamespace(model_name='alexnet',
                        identifier='test')
    D = SimpleNamespace()
    T = SimpleNamespace(batch_size=512,
                        num_workers=8)
    CFG = SimpleNamespace(M=M, D=D, T=T)

    from utils import complete_config
    model_dir='/home/tonglab/Desktop/test_training'
    CFG, model = complete_config(CFG, model_dir=model_dir)

    train_model(CFG)
