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

    architecture = 'blt',  # used to load architecture and name of top-level results directory
    architecture_args = SimpleNamespace(model='blt_blt', num_classes=1000, recurrent_steps=8, num_layers=4),  # key word args when initializing the model

    # cornet_s_custom params
    #architecture_args = dict(
    #    R = (1,2,4,2),          # recurrence, default = (1,2,4,2),
    #    K = (3,3,3,3),          # kernel size, default = (3,3,3,3),
    #    F = (64,128,256,512),   # feature maps, default = (64,128,256,512)
    #    S = 4,                  # feature maps scaling, default = 4
    #    out_channels = 1,       # number of heads, default = 1
    #    head_depth = 3,         # multi-layer head, default = 1
    #    head_width = 5,         # gridsize of adaptive avgpool layer,
    #    default = 1
    #),
    
    # cornet_st/flab parameters
    #architecture_args = dict(
    #    kernel_size = (3, 3, 3, 3),                          # kernel size,
    #    default = (3,3,3,3),
    #    num_features = (64,128,256,512)'                     # feature maps, default = (64,128,256,512)
    #    times = 2'
    #    out_channels = 1'  # number of heads, default = 1
    #    head_depth = 1'  # multi-layer head, default = 1
    #),

    identifier = 'blt_blt',  # name of second-level results directory
    model_dir = None, #op.expanduser('~/david/projects/p022_occlusion/models/cornet_s_V1/v2_occ-beh'),  # manual override for results directory
    finetune = False,  # starting a fine-tuning regime from pretrained weights?
    starting_params = None, #'/home/tonglab/david/masterScripts/DNN/zoo/pretrained_weights/cornet_s-1d3f7974.pth',  # starting params for fine-tuning
    finetune_dir = 'finetune_unocc',  # third-level directory for finetuned model results
    freeze_modules = ['V1', 'V2', 'V4','IT'],  # freeze the weights of these modules during finetuning
    return_model = False,  # return model object to environment after training
)

# dataset
D = SimpleNamespace(
    dataset = 'ILSVRC2012',
    num_views = 1,  # number of views to generate per example, default = 1
    transform_type = 'contrastive',  # 'contrastive' or 'default'
    cutmix = False,
    cutmix_args = dict(
        prob=1.0,
        alpha=1,
        beta=1,
        frgrnd=True),
)
"""
D.Occlusion = SimpleNamespace(
    form = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 
            'polkadot', 'polkasquare', 'crossBarOblique', 
            'crossBarCardinal', 'naturalUntexturedCropped2'],       # occluder type or list thereof
    probability = .8,                                               # probability that each image will be occluded, range(0,1)
    visibility = [.1, .2, .4, .6, .8],                              # image visibility or list thereof, range(0,1)
    colour = 'random',                                               # occluder colours (RGB uint8 / 'random' or list thereof. Ignored for 'textured' forms)
    views = [0],  # views to apply occlusion to, e.g. [0,2] will apply occluders to 1st and 3rd views
)
"""
# optimization
O = SimpleNamespace(
    num_epochs = 100,  # number of epochs to train for
    batch_size = 128,  # minibatch size
    optimizer = 'AdamW',  # any optimizer from torch.optim
    optimizer_args = dict(
        lr=0.0005,  # set float or one of the following strings: 'LRfinder', 'batch_linear', 'batch_nonlinear'
        weight_decay=1e-4,
        #momentum=.9,
    ),
    overwrite_optimizer=False,
    scheduler = 'StepLR',  # any attribute of torch.optim.lr_scheduler
    scheduler_args = dict(
        #patience = 8,
        gamma=.1,
        step_size = 100,
    ),
    criteria = dict(CrossEntropyLoss=dict(views=[0],weight=1)),  # loss function and, if multiple views per image, which views to apply loss to
    save_interval = 8,  # preserve params at every n epochs
    checkpoint = None,  # integer, resume training from this epoch (default: most recent, -1: from scratch)
    swa = None, # dict(start=8, anneal_epochs=4, swa_lr=.05)  # type: dict
    amp = False,  # automatic mixed precision. Can speed up optimization but can also cause over/underflow
)

CFG = SimpleNamespace(M=M,D=D,O=O)

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