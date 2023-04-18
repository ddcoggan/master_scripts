import sys
import pickle as pkl
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import numpy as np
import glob
import os
import os.path as op
from PIL import Image
import datetime
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import Namespace
import math
import pprint
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
sys.path.append(op.expanduser(f'~/david/master_scripts/DNN'))
from utils import accuracy

#torch.autograd.set_detect_anomaly(True)

def train_model(CFG):

    # unpack config
    M, D, T = CFG.M, CFG.D, CFG.T

    # image processing
    from utils import get_transforms
    train_path = op.join(f'{op.expanduser("~")}/Datasets/{D.dataset}', 'train')
    val_path = op.join(f'{op.expanduser("~")}/Datasets/{D.dataset}', 'val')
    transform_train, transform_val = get_transforms(D, T)
    train_data = ImageFolder(train_path, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=T.batch_size, shuffle=True, num_workers=T.workers)
    val_data = ImageFolder(val_path, transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=T.batch_size, shuffle=True, num_workers=T.workers)


    # loss functions
    if hasattr(T, 'device'):
        device = T.device
    else:
        from utils import configure_hardware
        T, device = configure_hardware(T)
    if 'classification' in T.learning:
        loss_class = nn.CrossEntropyLoss().to(device)  # send to device before constructing optimizer
    if 'contrastive' in T.learning:
        from utils import ContrastiveLoss
        loss_contr = ContrastiveLoss().to(device)  # can do both supervised and unsupervised contrastive loss


    # performance metrics
    metrics = []
    if 'classification' in T.learning:
        metrics += ['acc1', 'acc5', 'loss_class']
        loss_metric = 'loss_class'
    if 'contrastive' in T.learning:
        metrics += ['loss_contr']
        loss_metric = 'loss_contr'


    # log file
    logfile = f'{M.model_dir}/log.csv'  # stores batch-wise log of various metrics
    if not op.isfile(logfile):
        log_new_row_dict = {'epoch': [],
                            'batch': [],
                            'train_eval': []}
        for metric in metrics:
            for batch_epoch in ['batch', 'epoch']:
                log_new_row_dict[f'{metric}_{batch_epoch}'] = []
        log = pd.DataFrame(log_new_row_dict)
        log.to_csv(logfile, index=False)
    else:
        log = pd.read_csv(logfile)


    # epoch-wise stats file
    epoch_stats_path = f'{M.model_dir}/epoch_stats.pkl'
    if not op.isfile(epoch_stats_path):
        epoch_stats = {'train': {}, 'eval': {}}
        for metric in metrics:
            epoch_stats['train'][metric] = {}
            epoch_stats['eval'][metric] = {}
        pkl.dump(epoch_stats, open(epoch_stats_path, 'wb'))
    else:
        epoch_stats = pkl.load(open(epoch_stats_path, 'rb'))
        if 'loss' in epoch_stats['train']:
            for train_eval in epoch_stats:
                epoch_stats[train_eval][loss_metric] = epoch_stats[train_eval]['loss']
                epoch_stats[train_eval].pop('loss')
            pkl.dump(epoch_stats, open(epoch_stats_path, 'wb'))

    # parameters directory
    params_dir = op.join(M.model_dir, 'params')
    os.makedirs(params_dir, exist_ok=True)


    # model

    # put model on device
    if hasattr(M, 'model'):
        model = M.model
    else:
        from utils import get_model
        model = get_model(M)
    if T.nGPUs > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print(model)

    if T.last_epoch is None:

        print("Initialising new model state")

        # set starting point
        last_epoch = 0 # for optimizer
        next_epoch = 1
        train_evals = ['train', 'eval']

    else:

        print("Loading previous model state")

        last_epoch = T.last_epoch  # for optimizer

        # if interrupted during eval of last epoch, finish this before continuing
        if T.last_epoch not in epoch_stats['eval'][loss_metric] and T.last_epoch > 0:
            next_epoch = T.last_epoch
            train_evals = ['eval']
        else:
            next_epoch = T.last_epoch + 1
            train_evals = ['train', 'eval']

        # load model parameters
        params_path = f'{params_dir}/{T.last_epoch:03}.pt'
        params = torch.load(params_path)
        from utils import load_params
        model = load_params(params, model=model)

        # freeze weights if transfer learning
        if T.freeze_weights:
            for p, param in enumerate(model.parameters()):
                if p > T.freeze_weights:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # view param items
            # for p, param in enumerate(model.parameters()):
            #    print(f'item {p}, shape = {param.shape}, requires_grad = {param.requires_grad}')


    # optimizer

    if M.model_name.startswith('PredNet'):
        convparams = [p for p in model.baseconv.parameters()] + \
                     [p for p in model.FFconv.parameters()] + \
                     [p for p in model.FBconv.parameters()] + \
                     [p for p in model.linear.parameters()] + \
                     [p for p in model.GN.parameters()]
        rateparams = [p for p in model.a0.parameters()] + \
                     [p for p in model.b0.parameters()]
        if T.optimizer_name == 'SGD':
            optimizer = optim.SGD([{'params': convparams}, {'params': rateparams, 'weight_decay': 0}],
                                  lr=T.learning_rate,
                                  momentum=.9,
                                  weight_decay=.0001)
    else:
        if T.optimizer_name == 'SGD':
            optimizer = optim.SGD(params=model.parameters(), lr=T.learning_rate, momentum=T.momentum)
            optimizer.param_groups[0]['initial_lr'] = T.learning_rate
        elif T.optimizer_name == 'Adam':
            optimizer = optim.Adam(params=model.parameters(), lr=T.learning_rate)

    # load optimizer state unless starting afresh for transfer learning
    if T.last_epoch is None or (T.last_epoch == 0 and T.freeze_weights):
        print('Initialising new optimizer state')
    else:
        print('Loading previous optimizer state')
        from utils import load_params
        optimizer = load_params(params, optimizer=optimizer)

    # scheduler to adapt optimizer parameters throughout training
    if T.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=T.step_size, last_epoch=last_epoch)
    if T.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    # save initial model and optimizer parameters
    if T.last_epoch is None:
        epoch_save_path = op.join(params_dir, '000.pt')
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
        torch.save(params, epoch_save_path)


    ### TRAIN ###
    # NOTE: epoch 0 (where initial weights are evaluated on test and val datasets) performed last so learning starts immediately

    for epoch in list(range(next_epoch, T.num_epochs+1)) + [0]:

        # load epoch 0 params at the end
        if epoch == 0:
            from utils import load_params
            params_path = f'{params_dir}/000.pt'
            model = load_params(params_path, model=model)

        for train_eval in train_evals:

            # train/eval specific settings
            if train_eval == 'train':
                loader = train_loader
                log_string = 'Training'.ljust(10)
                model.train() if epoch != 0 else model.eval()
            else:
                loader = val_loader
                log_string = 'Evaluating'
                model.eval()

            # initialize cumulative performance stats for this epoch
            for metric in metrics:
                epoch_stats[train_eval][metric][epoch] = 0

            # loop through batches
            with tqdm(loader, unit=f"batch({T.batch_size})") as tepoch:

                for batch, (inputs, targets) in enumerate(tepoch):

                    tepoch.set_description(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {log_string} | Epoch {epoch}/{T.num_epochs}')

                    if train_eval == 'train' and epoch > 0:
                        optimizer.zero_grad(set_to_none=True) # set_to_none saves even more memory1

                    # initialize performance stats for this batch
                    batch_stats = {}

                    # for contrastive learning, alter images, perform remaining image transform
                    if 'contrastive' in T.learning:
                        from utils import AlterImages, get_remaining_transform
                        inputs = AlterImages(D,T)(inputs)
                        remaining_transform = get_remaining_transform(train_eval)
                        inputs = remaining_transform(inputs)

                    # save some input images
                    if epoch == 1 and batch == 0:
                        from utils import save_image_custom
                        sample_input_dir = f'{M.model_dir}/sample_inputs'
                        os.makedirs(sample_input_dir, exist_ok=True)
                        save_image_custom(inputs, T, sample_input_dir, max_images=32)

                    # put inputs on device
                    non_blocking = 'contrastive' in T.learning
                    inputs = inputs.to(device, non_blocking=non_blocking)
                    targets = targets.to(device, non_blocking=non_blocking)

                    # pass through model
                    outputs = model(inputs)

                    # separate by classification/contrastive
                    if M.model_name in ['cornet_s_custom','cornet_st'] and M.out_channels == 2:
                        outputs_class = outputs[:,:,0]
                        outputs_contr = outputs[:,:,1]
                        if 'contrastive' in T.learning:
                            targets_class = torch.cat([targets, targets], dim=0)
                        else:
                            targets_class = targets
                    elif T.learning == 'supervised_classification':
                        outputs_class = outputs
                        targets_class = targets
                    else:
                        outputs_contr = outputs

                    # classification accuracy and loss
                    if 'classification' in T.learning:
                        batch_stats['acc1'], batch_stats['acc5'] = [x.detach().cpu() for x in accuracy(outputs_class, targets_class, (1,5))]
                        epoch_stats[train_eval]['acc1'][epoch] = ((epoch_stats[train_eval]['acc1'][epoch] * batch) + batch_stats['acc1']) / (batch + 1)
                        epoch_stats[train_eval]['acc5'][epoch] = ((epoch_stats[train_eval]['acc5'][epoch] * batch) + batch_stats['acc5']) / (batch + 1)
                        loss_cl = loss_class(outputs_class, targets_class)  # leave this on gpu for back prop
                        batch_stats['loss_class'] = loss_cl.detach().cpu().item()  # store copy of loss evalue on cpu
                        epoch_stats[train_eval]['loss_class'][epoch] = ((epoch_stats[train_eval]['loss_class'][epoch] * batch) + batch_stats['loss_class']) / (batch + 1)

                    # contrastive accuracy and loss
                    if 'contrastive' in T.learning:
                        features = torch.stack(torch.split(outputs_contr, [targets.shape[0], targets.shape[0]], dim=0), dim=1) # unstack and combine along new (contrastive) dimension
                        if 'unsupervised_contrastive' in T.learning:
                            loss_co = loss_contr(features)  # leave this on gpu for back prop
                            batch_stats['loss_contr'] = loss_co.detach().cpu().item()  # store copy of loss value on cpu
                        elif 'supervised_contrastive' in T.learning:
                            loss_co = loss_contr(features, targets)  # leave this on gpu for back prop
                            batch_stats['loss_contr'] = loss_co.detach().cpu().item()  # store copy of loss value on cpu
                        epoch_stats[train_eval]['loss_contr'][epoch] = ((epoch_stats[train_eval]['loss_contr'][epoch] * batch) + batch_stats['loss_contr']) / (batch + 1)

                    # display performance metrics for this batch
                    postfix_string = ''
                    for metric in metrics:
                        postfix_string += f"{metric}={batch_stats[metric]:.4f}({epoch_stats[train_eval][metric][epoch]:.4f}) | "
                    postfix_string += f"lr={optimizer.param_groups[0]['lr']*(train_eval == 'train' and epoch != 0):.5f}"
                    tepoch.set_postfix_str(postfix_string)

                    # log performance metrics for this batch
                    log_new_row_dict = {'epoch': epoch,
                                        'batch': batch,
                                        'train_eval': train_eval}
                    for metric in metrics:
                        log_new_row_dict[f'{metric}_batch'] = batch_stats[metric]
                        log_new_row_dict[f'{metric}_epoch'] = epoch_stats[train_eval][metric][epoch]
                    log_new_row = pd.DataFrame(log_new_row_dict, index=[0])
                    log = pd.concat([log, log_new_row])

                    # compute gradients and optimize parameters
                    if train_eval == 'train' and epoch > 0:
                        if M.model_name in ['cornet_s_custom','cornet_st'] and M.out_channels == 2:
                            if hasattr(t, 'loss_ratio'):
                                loss_unit = 1 / sum(T.loss_ratio)
                                loss_cl_weight, loss_co_weight = [T.loss_ratio[l] * loss_unit for l in range(len(T.loss_ratio))]
                            else:
                                loss_cl_weight, loss_co_weight = (.5,.5)
                            loss = loss_cl*loss_cl_weight + loss_co*loss_co_weight
                        elif 'classification' in T.learning:
                            loss = loss_cl
                        else:
                            loss = loss_co
                        loss.backward()
                        optimizer.step()

            # update LR scheduler
            if epoch > 0:
                if T.scheduler == 'StepLR' and train_eval == 'train':
                    scheduler.step()
                elif T.scheduler == 'ReduceLROnPlateau' and train_eval == 'eval':
                    scheduler.step(epoch_stats['eval'][loss_metric][epoch])

            # ensure next epoch does both train and eval
            train_evals = ['train', 'eval']

            if train_eval == 'train' and epoch > 0:

                # save new model/optimizer/params state
                epoch_save_path = f"{params_dir}/{epoch:03}.pt"
                params = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
                torch.save(params, epoch_save_path)

                # delete old model and optimizer state
                if epoch > 1 and (epoch-1) % M.save_interval != 0:
                    last_save_path = f"{params_dir}/{epoch-1:03}.pt"
                    os.remove(last_save_path)

            # save log file and epoch stats
            log.to_csv(logfile, index=False)
            pkl.dump(epoch_stats, open(epoch_stats_path, 'wb'))

            # make plots
            plotdir = op.join(M.model_dir, 'plots')  # stores plots of various metrics
            os.makedirs(plotdir, exist_ok=True)
            for metric in metrics:
                epochs_train = sorted(epoch_stats['train'][metric].keys())
                epochs_eval = sorted(epoch_stats['eval'][metric].keys())
                plt.plot(epochs_train, [epoch_stats['train'][metric][epoch] for epoch in epochs_train], label='train')
                plt.plot(epochs_eval, [epoch_stats['eval'][metric][epoch] for epoch in epochs_eval], label='eval')
                plt.xlabel('epoch')
                plt.ylabel(metric)
                if metric == 'loss_contr':
                    plt.yscale('log')
                plt.legend()
                plt.grid(True)
                plt.savefig(op.join(plotdir, f'{metric}.png'))
                plt.show()
                plt.close()
        
    print('Training complete.')

    if hasattr(M, 'return_model') and M.return_model:
        return model



if __name__ == '__main__':

    """
    configure a model for training
    """

    # often used occluders and visibility levels
    occluders_fMRI = ['barHorz04', 'barVert12', 'barHorz08']
    occluders_behavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot', 'polkasquare',
                             'crossBarOblique', 'crossBarCardinal', 'naturalUntexturedCropped2']
    visibilities = [.1, .2, .4, .8]


    class CFG:
        # model
        class M:
            model_name = 'cornet_s'
            identifier = 'classification_unaltered'  # used to name model directory, required
            save_interval = 4  # preserve params at every n epochs
            return_model = False  # return model object to environment after training
            # init_params = ['contrastive_random_occluder', 32]   # starting params model and epoch (e.g. when starting transfer learning)
            params_path = '/home/tonglab/david/masterScripts/DNN/zoo/pretrained_weights/cornet_s-1d3f7974.pth'
            """
            # cornet_s_custom parameters
            model_name = 'cornet_s_custom'                             # used to load architecture, required
            R = (1,2,4,2)                                       # recurrence, default = (1,2,4,2),
            K = (3,3,3,3)                                       # kernel size, default = (3,3,3,3),
            F = (64,128,256,512)                                # feature maps, default = (64,128,256,512)
            S = 4                                               # feature maps scaling, default = 4
            out_channels = 1                                    # number of heads, default = 1
            head_depth = 1                                     # multi-layer head, default = 1

            # cornet_st/flab parameters
            model_name = 'cornet_flab'  # used to load architecture, required
            kernel_size = (3, 3, 3, 3)                          # kernel size, default = (3,3,3,3),
            num_features = (64,128,256,512)                     # feature maps, default = (64,128,256,512)
            times = 2
            out_channels = 1  # number of heads, default = 1
            head_depth = 1  # multi-layer head, default = 1
            """

        # dataset
        class D:
            dataset = 'ILSVRC2012'
            contrast = 'occluder_translate'  # contrastive learning manipulation. Options: 'repeat_transform','occluder_translate'
            transform = 'standard'
            """
            class Occlusion:
                type = occluders_behavioural                       # occluder type or list thereof
                prop_occluded = .8                                 # proportion of images to be occluded
                visibility = visibilities                          # image visibility or list thereof, range(0,1)
                colour = [(0,0,0),(127,127,127),(255,255,255)]      # occluder colours (unless naturalTextured type)

            class Blur:
                sigmas = [0,1,2,4,8]
                weights = [1,1,1,1,1]

            class Noise:
                type = 'gaussian'
                ssnr = [.1,.2,.4,.8,1]
                label = 'mixedNoise'
            """

        # training
        class T:
            optimizer_name = 'SGD'  # SGD or ADAM
            batch_size = 64  # calculated if not set
            # learning_rate = 2 ** -7                 # calculated if not set
            force_lr = False  # by default, learning rate continues existing schedule. Set to True to override this behaviour
            momentum = .9
            scheduler = 'StepLR'
            step_size = 16
            num_epochs = 32  # number of epochs to train for
            learning = 'supervised_classification'  # supervised_classification, supervised_contrastive, self-supervised_contrastive
            freeze_weights = None  # freeze params up to this index (for transfer learning) cornet_s = 84
            nGPUs = -1  # if not set, or set to -1, all GPUs visible to pytorch will be used. Set to 0 to use CPU
            GPUids = 1  # list of GPUs to use (ignored if nGPUs in [-1,0] or not set)
            workers = 2  # number of CPU threads to use (calculated if not set)
            last_epoch = None  # resume training from this epoch (set to None to use most recent)

    train_model(CFG)
