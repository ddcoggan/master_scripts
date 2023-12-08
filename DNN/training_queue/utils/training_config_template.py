"""
configure a model for training
"""
import os
import os.path as op
import glob
from types import SimpleNamespace
import pickle as pkl

# model
M = SimpleNamespace(

    # mandatory
    model_name = 'cornet_s_hw3',
    identifier = 'standard_training',

    # optional
    save_interval = 4,  # preserve params at every n epochs
    transfer = True,
    transfer_dir = 'transfer_task-cont',
    freeze_layers = ['V1', 'V2', 'V4', 'IT'],
    #starting_params = '/home/tonglab/david/masterScripts/DNN/zoo/pretrained_weights/cornet_s-1d3f7974.pth',
)
"""
# cornet_s_custom parameters
M.R = (1,2,4,2)                           # recurrence, default = (1,2,4,2),
M.K = (3,3,3,3)                           # kernel size, default = (3,3,3,3),
M.F = (64,128,256,512)                    # feature maps, default = (64,128,256,512)
M.S = 4                                   # feature maps scaling, default = 4
M.out_channels = 1                        # number of heads, default = 1
M.head_depth = 3                          # multi-layer head, default = 1
M.head_width = 3                          # head width, default = 1

# cornet_st/flab parameters
M.kernel_size = (3, 3, 3, 3)                          # kernel size, default = (3,3,3,3),
M.num_features = (64,128,256,512)                     # feature maps, default = (64,128,256,512)
M.times = 2
M.out_channels = 1  # number of heads, default = 1
M.head_depth = 1  # multi-layer head, default = 1
"""

# dataset
D = SimpleNamespace(
    dataset = 'ILSVRC2012',
    transform_type = 'contrastive'  # 'contrastive' or 'default'
    num_views = 2,  # number of views to generate per example
    views_occluded = [1],  # indices of views to apply alterations to
)
"""
occluders_behavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot', 'polkasquare',
                         'crossBarOblique', 'crossBarCardinal', 'naturalUntexturedCropped2']
visibilities = [.1, .2, .4, .6, .8]
D.Occlusion = SimpleNamespace(
    type = occluders_behavioural,                     # occluder type or list thereof
    prop_occluded = .8,                                 # proportion of images to be occluded
    visibility = [.6,.7,.8,.9],                          # image visibility or list thereof, range(0,1)
    colour = ['random']      # occluder colours (unless naturalTextured type)
)
"""

# training
T = SimpleNamespace(
    num_epochs = 43,  # number of epochs to train for
    optimizer_name = 'SGD',  # SGD or ADAM
    batch_size = 256,
    learning_rate = .1,
    weight_decay = 1e-4,
    #overwrite_optimizer = False,
    momentum = .9,
    scheduler = 'StepLR',
    #patience = 5,
    step_size = 20,
    #num_workers = 14,
    classification = False,
    #views_class = [1],  # views to apply classification loss
    contrastive = True,
    #views_contr = [0,1],  # views to apply contrastive loss
    contrastive_supervised = False,
    checkpoint = None,  # resume training from this epoch (set to None or don't set to use most recent)
    SWA = False,
    SWA_start = 8,
    SWA_freq = 4,
    SWA_lr = .05,
    cutmix = False,
    cutmix_prob = 1.0,
    cutmix_alpha = 1,
    cutmix_beta = 1,
    cutmix_frgrnd = True,
    AMP = False,
)

CFG = SimpleNamespace(M=M,D=D,T=T)

if __name__ == "__main__":

    # complete configuration
    from complete_config import complete_config
    CFG = complete_config(CFG, resolve='resume')

    # output directory
    if not hasattr(CFG.M, 'model_dir'):
        CFG.M.model_dir = op.expanduser(f'~/david/projects/p022_occlusion/in_silico/models/{CFG.M.model_name}/{CFG.M.identifier}')

    import sys
    sys.path.append(op.expanduser('~/david/master_scripts/DNN/utils'))

    from train_model import train_model
    train_model(CFG)
