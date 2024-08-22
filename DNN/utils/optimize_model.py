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
from utils import (accuracy, get_outputs, AverageMeter, cutmix,
                   plot_performance, save_image_custom, get_loaders,
                   get_criteria)

torch.random.manual_seed(42)

def now():
    return datetime.now().strftime("%y/%m/%d %H:%M:%S")

def optimize_model(CFG, model=None, verbose=False):

    # unpack config
    M, D, O = CFG.M, CFG.D, CFG.O

    # initialise model directory and save config
    os.makedirs(f'{M.model_dir}/params', exist_ok=True)
    pkl.dump(CFG, open(f'{CFG.M.model_dir}/config.pkl', 'wb'))
    from utils import config_to_text
    config_to_text(CFG)

    # configure hardware
    num_GPUs = torch.cuda.device_count()
    device = torch.device('cuda') if num_GPUs else torch.device('cpu')
    num_workers = num_GPUs * 8 if os.cpu_count() <= 32 else num_GPUs * 16
    
    # image processing
    loader_train, loader_val = get_loaders(D, O, num_workers)

    # loss functions (send to device before constructing optimizer)
    criteria = get_criteria(O.criterion, device)

    # performance metrics
    metrics = []
    for criterion, metric_set in criteria.items():
        for metric, value in metric_set.items():
            if metric != 'func' and not metric.startswith('sched'):
                metrics.append(metric)
    sched_criteria = criteria[sorted(criteria.keys())[-1]]

    # performance table
    performance_path = f'{M.model_dir}/performance.csv'
    if op.isfile(performance_path):
        performance = pd.read_csv(open(performance_path, 'r+'), index_col=0)
        step = performance['step'].max()
    else:
        performance = pd.DataFrame()
        step = 0

    # model
    if model is None:
        from utils import get_model
        model = get_model(M.architecture, M.architecture_args)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    if verbose:
        print(model)
        print(f'model parameters: {num_parameters}')
        print(f'workers: {num_workers}')
        print(f'metrics: {metrics}')
        

    # find model params
    if O.checkpoint == -1:
        params_path = M.starting_params if \
            hasattr(M, 'starting_params') else None
        modules = M.freeze_layers if hasattr(M, 'freeze_layers') else 'all'
    else:
        params_path = f'{M.model_dir}/params/{O.checkpoint:03}.pt'
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
    optimizer = getattr(torch.optim, O.optimizer)(
        params=model.parameters(), **O.optimizer_args)
    scheduler = getattr(torch.optim.lr_scheduler, O.scheduler)(
        optimizer, **O.scheduler_args)
    if O.swa is not None:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, last_epoch=O.checkpoint, **O.swa)
        
    # load optimizer and scheduler states
    if O.checkpoint > -1:
        print('Loading optimizer state')
        optimizer = load_params(params, optimizer, 'optimizer')


    # save initial model and optimizer state
    if O.checkpoint == -1:
        print('Saving initial model and optimizer states')
        params_path = f'{M.model_dir}/params/000.pt'
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
        if O.swa:
            params['swa_model'] = swa_model.state_dict()
        torch.save(params, params_path)

    # speed up training with mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=O.amp)

    # define train / eval loop
    def one_epoch(model, M, D, O, loader, optimizer, scaler, metrics,
                  train_eval, performance, criteria, step):

        # initialize cumulative performance stats for this epoch and batch set
        performance_epoch = {metric: AverageMeter() for metric in metrics}
        performance_batch = {metric: AverageMeter() for metric in metrics}

        # loop through batches
        with tqdm(loader, unit=f"batch({O.batch_size})") as tepoch:

            for batch, (inputs, targets) in enumerate(tepoch):


                tepoch.set_description(f'{now()} | {train_eval.ljust(5)} | epoch'
                                       f' {epoch}/{O.num_epochs}')

                # save some input images after one successful batch
                if epoch == 1 and batch == 32:
                    save_image_custom(
                        inputs, f'{M.model_dir}/sample_{train_eval}_inputs')

                # concatenate multiple views along image dimension
                inputs = torch.concat(inputs, dim=0).squeeze()

                # put inputs on device
                non_blocking = any(['SimCLRLoss' in c for c in criteria])
                inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)

                # automatic mixed precision
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16, enabled=O.amp):

                    # pass through model
                    outputs = model(inputs)

                    # losses
                    for criterion in criteria:
                        
                        # get loss
                        relevant_outputs = get_outputs(outputs, M, D, O, criterion)
                        relevant_targets = None if criterion == 'SimCLRLoss' else targets
                        criteria[criterion][criterion] = criteria[criterion]['func'](relevant_outputs, relevant_targets)
                        
                        # accuracy for cross entropy loss
                        if criterion == 'CrossEntropyLoss':
                            trg = torch.concat([targets] * D.num_views, dim=0)
                            acc1, acc5 = [x.detach().cpu() for x in accuracy(relevant_outputs, trg, (1, 5))]
                            performance_batch['acc1'].update(acc1)
                            performance_batch['acc5'].update(acc5)
                            
                            # if cutmix, combine background target loss with foreground target loss
                            if D.cutmix and apply_cutmix and not D.cutmix_args['frgrnd']:
                                trg = torch.concat([trg_frgrnd] * D.num_views, dim=0)
                                criteria[criterion][criterion] += criteria[criterion]['func'](relevant_outputs, trg) * lam
                        performance_batch[criterion].update(criteria[criterion][criterion].clone().detach().cpu().item())

                # display performance metrics for this batch
                postfix_string = ''
                for metric, value in performance_batch.items():
                    postfix_string += (
                        f"{metric}={value.val:.4f}({value.avg:.4f}) ")
                lr = optimizer.param_groups[0]["lr"] if train_eval == 'train' else 0
                tepoch.set_postfix_str(postfix_string + f'lr={lr:.5f}')
                
                # update model
                if train_eval == 'train':
                    loss_values = torch.tensor([criteria[c][c] for c in
                                                criteria], device=device, 
                                               requires_grad=True)
                    loss_weights = torch.tensor([O.criteria[c]['weight'] for 
                                                 c in criteria], device=device)
                    loss = torch.sum(loss_values * loss_weights)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                # every 1024 steps, save performance and plot
                if step % 1024 == 0 or batch == len(loader) - 1:
                    performance = pd.concat([performance, pd.DataFrame({
                        **{'time': [now()], 'step': [step], 'epoch': [epoch],
                        'train_eval': [train_eval], 'lr': [lr]},
                        **{key: np.array(value.avg, dtype="float16")
                            for key, value in (performance_batch.items())}})])
                    performance = (performance.sort_values(
                        by=['step', 'train_eval'], ascending=[True, False])
                        .reset_index(drop=True))
                    plot_performance(performance, metrics, M.model_dir)
                    for m, v in performance_batch.items():
                        performance_epoch[m].update(v.avg)
                        v.reset()

        return model, optimizer, scaler, performance, step


    # train / eval loop
    for epoch in list(range(max(1, O.checkpoint), O.num_epochs+1)):

        # train
        model.train()
        model, optimizer, scaler, performance, step = one_epoch(
            model, M, D, O, loader_train, optimizer, scaler, metrics, 'train', performance, criteria, step)

        # update LR scheduler and perform SWA
        if O.swa and epoch >= O.swa_args['start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if epoch == O.num_epochs:
                update_bn(train_loader, swa_model)
        else:
            if O.scheduler == 'StepLR':
                scheduler.step()

        # save new state
        params_path = f"{M.model_dir}/params/{epoch:03}.pt"
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
        if O.swa:
            params['swa_model'] = swa_model.state_dict()
        torch.save(params, params_path)

        # delete previous state
        if not hasattr(O, 'save_interval') or (
                epoch > 1 and O.save_interval > 1 and
                (epoch - 1) % O.save_interval):
            last_save_path = f"{M.model_dir}/params/{epoch - 1:03}.pt"
            if op.exists(last_save_path):
                os.remove(last_save_path)

        # eval
        model.eval()
        with torch.no_grad():
            model, optimizer, scaler, performance = one_epoch(
                model, M, D, O, loader_val, optimizer, scaler, metrics, 'eval',
                performance, criteria, step)

        # update schedulers that use eval performance
        if not O.swa and O.scheduler == 'ReduceLROnPlateau':
            scheduler.step(performance[[sched_criteria['sched_metric']]].values[-1])
            torch.save(params, params_path)

        # save if best eval performance
        prev = performance[sched_metric][
            performance['train_eval'] == 'eval'].to_list()
        new = prev.pop()
        if sched_criteria['sched_compare'](new, prev).all():
            for prev_best in glob.glob(f'{M.model_dir}/params/best*.pt'):
                os.remove(prev_best)
            torch.save(params, f'{M.model_dir}/params/best_{epoch:03}.pt')
             
        # plot performance
        plot_performance(performance, metrics, M.model_dir)
        performance.to_csv(performance_path, index=False)
        
        torch.cuda.empty_cache()

    print('Training complete.')
    
  


if __name__ == '__main__':

    M = SimpleNamespace(architecture='alexnet',
                        identifier='test')
    D = SimpleNamespace()
    O = SimpleNamespace(batch_size=1024,
                        num_workers=8,
                        AMP=False)
    CFG = SimpleNamespace(M=M, D=D, O=T)

    from utils import complete_config
    model_dir = '/home/tonglab/Desktop/test_training'
    CFG, model = complete_config(CFG, model_dir=model_dir)

    train_model(CFG, model=model)
