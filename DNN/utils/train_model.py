import sys
import pickle as pkl
import pandas as pd
import numpy as np
import glob
import os
import os.path as op
from PIL import Image
from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace
import math
import pprint
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd.variable import Variable
import torch.nn.functional as F
import copy
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from accuracy import accuracy
from assign_outputs import assign_outputs
from AverageMeter import AverageMeter
from cutmix import cutmix
from plot_performance import plot_performance
from save_image_custom import save_image_custom


#torch.backends.cudnn.benchmark = True
#torch.autograd.set_detect_anomaly(True)
torch.random.manual_seed(42)

def now():
    return datetime.now().strftime("%y/%m/%d %H:%M:%S")

def train_model(CFG, model=None, verbose=False):

    ## force cuda initialization
    #torch.nn.functional.conv2d(
    #    torch.zeros(32, 32, 32, 32, device=torch.device('cuda')),
    #    torch.zeros(32, 32, 32, 32, device=torch.device('cuda')))

    # unpack config
    M, D, T = CFG.M, CFG.D, CFG.T

    # initialise model directory and save config
    os.makedirs(f'{M.model_dir}/params', exist_ok=True)
    pkl.dump(CFG, open(f'{CFG.M.model_dir}/config.pkl', 'wb'))
    from utils import config_to_text
    config_to_text(CFG)

    # configure hardware
    num_GPUs = torch.cuda.device_count()
    device = torch.device('cuda') if num_GPUs else torch.device('cpu')
    num_workers = num_GPUs * 8 if os.cpu_count() <= 32 else num_GPUs * 8
    print(f'{num_workers} workers') if verbose else print()

    # image processing
    from utils import get_loaders
    loader_train, loader_val = get_loaders(D, T, num_workers)

    # loss functions (send to device before constructing optimizer)
    from utils import get_criteria
    criteria = get_criteria(T, device)

    # performance metrics
    metrics = []
    if T.classification:
        metrics += ['acc1', 'acc5', 'loss_class']
        loss_metric, sched_metric, sched_mode = 'loss_class', 'acc1', 'max'
    if T.contrastive:
        metrics += ['loss_contr']
        loss_metric, sched_metric, sched_mode = \
            'loss_contr', 'loss_contr', 'min'
    compare = np.less if sched_mode == 'min' else np.greater

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
        model = get_model(M.model_name, {'M': M})
    print(model) if verbose else print()

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
        params = torch.load(params_path, map_location=device)
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
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # optimizer
    from utils import get_optimizer
    optimizer = get_optimizer(model, T)
    if T.SWA:
        swa_model = AveragedModel(model)
    
    # scheduler
    from utils import get_scheduler
    scheduler = get_scheduler(optimizer, T)
    if T.SWA:
        swa_scheduler = SWALR(optimizer, anneal_epochs=T.SWA_freq,
                              swa_lr=T.SWA_lr, last_epoch=T.checkpoint)
        
    # load optimizer and scheduler states
    if T.checkpoint > -1:
        print('Loading optimizer state')
        optimizer = load_params(params, optimizer, 'optimizer')


    # save initial model and optimizer state
    if T.checkpoint == -1:
        print('Saving initial model and optimizer state')
        params_path = f'{M.model_dir}/params/000.pt'
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
        if T.SWA:
            params['swa_model'] = swa_model.state_dict()
        torch.save(params, params_path)

    # speed up training with mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=T.AMP)

    # define train / eval loop
    def one_epoch(model, M, D, T, loader, optimizer, scaler, metrics,
                  train_eval, performance, criteria):

        # initialize cumulative performance stats for this epoch
        epoch_tracker = {metric: AverageMeter() for metric in metrics}

        # loop through batches
        with tqdm(loader, unit=f"batch({T.batch_size})") as tepoch:

            for batch, (inputs, targets) in enumerate(tepoch):


                tepoch.set_description(f'{now()} | {train_eval.ljust(5)} | epoch'
                                       f' {epoch}/{T.num_epochs}')

                # save some input images after one successful batch
                if epoch == 1 and batch == 32:
                    save_image_custom(
                        inputs, f'{M.model_dir}/sample_{train_eval}_inputs')

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
                        targets_contr = targets if hasattr(T, 'contrastive_supervised') and \
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
                lr = optimizer.param_groups[0]["lr"] if train_eval == 'train' else 0
                tepoch.set_postfix_str(postfix_string + f'lr={lr:.5f}')

                # update model
                if train_eval == 'train':
                    if not (T.classification and T.contrastive):
                        loss = loss_class if T.classification else loss_contr
                    else:
                        if T.loss_ratio:
                            loss_ratio_norm = F.normalize(torch.tensor(
                                T.loss_ratio).float(), p=1, dim=0)
                        else:
                            loss_ratio_norm = (.5, .5)
                        loss = loss_class * loss_ratio_norm[0] + loss_contr * \
                               loss_ratio_norm[1]

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        # add performance for this epoch
        new_stats = {'time': [now()], 'epoch': [epoch],
                     'train_eval': [train_eval], 'lr': [lr]}
        new_stats = {**new_stats,
                     **{key: np.array(value.avg, dtype="float16")
                        for key, value in (epoch_tracker.items())}}
        performance = pd.concat(
            [performance, pd.DataFrame(new_stats)]).reset_index(drop=True)
        performance = performance.sort_values(
            by=['epoch', 'train_eval'], ascending=[True, False]).reset_index(
            drop=True)
        performance.to_csv(performance_path)

        torch.cuda.empty_cache()

        return model, optimizer, scaler, performance


    # train / eval loop
    for epoch in list(range(max(1, T.checkpoint), T.num_epochs+1)):

        # train
        model.train()
        model, optimizer, scaler, performance = one_epoch(
            model, M, D, T, loader_train, optimizer, scaler, metrics, 'train', performance, criteria)

        # update LR scheduler and perform SWA
        if T.SWA and epoch > T.SWA_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if epoch == T.num_epochs:
                update_bn(train_loader, swa_model)
        else:
            if T.scheduler == 'StepLR':
                scheduler.step()

        # save new state
        params_path = f"{M.model_dir}/params/{epoch:03}.pt"
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
        if T.SWA:
            params['swa_model'] = swa_model.state_dict()
        torch.save(params, params_path)

        # delete previous state
        if not hasattr(M, 'save_interval') or (
                epoch > 1 and M.save_interval > 1 and
                (epoch - 1) % M.save_interval):
            last_save_path = f"{M.model_dir}/params/{epoch - 1:03}.pt"
            if op.exists(last_save_path):
                os.remove(last_save_path)

        # eval
        model.eval()
        with torch.no_grad():
            model, optimizer, scaler, performance = one_epoch(
                model, M, D, T, loader_val, optimizer, scaler, metrics, 'eval',
                performance, criteria)

        # update schedulers that use eval performance
        if not T.SWA and T.scheduler == 'ReduceLROnPlateau':
            scheduler.step(performance[sched_metric].values[-1])
            params['scheduler'] = scheduler.state_dict()
            torch.save(params, params_path)

        # save if best eval performance
        prev = performance[sched_metric][
            performance['train_eval'] == 'eval'].to_list()
        new = prev.pop()
        if compare(new, prev).all():
            for prev_best in glob.glob(f'{M.model_dir}/params/best*.pt'):
                os.remove(prev_best)
            torch.save(params, f'{M.model_dir}/params/best_{epoch:03}.pt')
             
        # plot performance
        if epoch != 1:
            plot_performance(performance, metrics, M.model_dir)

        torch.cuda.empty_cache()

    print('Training complete.')
    
  


if __name__ == '__main__':

    M = SimpleNamespace(model_name='alexnet',
                        identifier='test')
    D = SimpleNamespace()
    T = SimpleNamespace(batch_size=1024,
                        num_workers=8,
                        AMP=False)
    CFG = SimpleNamespace(M=M, D=D, T=T)

    from utils import complete_config
    model_dir = '/home/tonglab/Desktop/test_training'
    CFG, model = complete_config(CFG, model_dir=model_dir)

    train_model(CFG, model=model)
