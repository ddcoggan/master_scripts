import glob
import torch
import torch.nn as nn
import os
import os.path as op
import numpy as np
import torchvision.transforms as transforms
from argparse import Namespace
import sys
from torch.utils.data import DataLoader
import shutil
from types import SimpleNamespace

sys.path.append(os.path.expanduser("~/david/master_scripts/DNN"))

def get_activations(M, image_dir, T=SimpleNamespace(), layers=None,
                    sampler=None,
                    norm_minibatch=False, save_input_samples=False, sample_input_dir=None, transform=None, shuffle=False):

    torch.no_grad()

    # hardware
    from utils import configure_hardware
    T, device = configure_hardware(T, verbose=True)

    # model
    if not hasattr(M, 'model'):
        print('loading model...')
        from utils import get_model
        model = get_model(M)
    else:
        model = M.model
    model.to(device)

    #print(model)


    # load params
    if (not hasattr(M, 'params_loaded') or M.params_loaded is False) and (not hasattr(M, 'pretrained') or M.pretrained is False):
        print('loading parameters...')
        from utils import load_params
        model = load_params(M.params_path, model=model)

    if norm_minibatch:
        model.train()  # forces batch norm layers to use minibatch stats
    else:
        model.eval()  # default uses stored running means and variances

    # calculate optimal batch size
    if not hasattr(T, 'batch_size'):
        from utils import calculate_batch_size
        T = calculate_batch_size(model, T, device)
        print(f'optimal batch size calculated at {T.batch_size}')

    # image transforms
    if transform is None:
        from utils import get_transforms
        _, transform = get_transforms()


    # image loader
    from utils import CustomDataSet
    dataset = CustomDataSet(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=shuffle, num_workers=T.num_workers, sampler=sampler, pin_memory=True)

    # define activation hook
    def get_activation(idx):
        def hook(model, input, output):
            if type(output) == torch.Tensor:
                activations[idx] = torch.concat([activations[idx], output.detach().cpu()])
            elif 'cornet_rt' in M.model_name and type(output) == tuple:
                activations[idx] = torch.concat([torch.tensor(activations[idx]), output[0].detach().cpu()])
            else:
                activations[idx].append(output)
        return hook

    # store activations in a dict with items for each layer
    activations = {}

    if not layers or 'input' in layers:
        activations['input'] = torch.empty(0)



    # add activation hook to model


    if M.model_name in ['alexnet', 'vgg19']:
        from utils import model_layer_labels
        layers = model_layer_labels[M.model_name]
        # set up activations dict
        for layer in layers:
            activations[layer] = torch.empty(0)

        # cycle through model registering forward hook
        layer_counter = 0
        for l, layer in enumerate(model.features):
            model.features[l].register_forward_hook(get_activation(layers[layer_counter]))
            layer_counter += 1
        model.avgpool.register_forward_hook(get_activation(layers[layer_counter]))
        layer_counter += 1
        for l, layer in enumerate(model.classifier):
            model.classifier[l].register_forward_hook(get_activation(layers[layer_counter]))
            layer_counter += 1

    elif M.model_name.startswith('PredNet'):

        output, (x_ff_Ctr, x_fb_Ctr, x_pred_Ctr, x_err_Ctr, x_ff_beforeGN_Ctr, x_fb_beforeGN_Ctr) = model(inputs)
        activation['feedforward'] = x_ff_Ctr
        activation['feedback'] = x_fb_Ctr
        activation['prediction'] = x_pred_Ctr
        activation['error'] = x_err_Ctr
        activation['feedforward_preGroupNorm'] = x_ff_beforeGN_Ctr
        activation['feedback_preGroupNorm'] = x_fb_beforeGN_Ctr


    elif M.model_name == 'cornet_st' or M.model_name.startswith('cornet_rt'):

        for l, layer in enumerate(['V1', 'V2', 'V4', 'IT']):
            if not layers or layer in layers:
                activations[layer] = []
                getattr(model, layer).register_forward_hook(get_activation(layer))

    elif M.model_name.startswith('cornet'):

        for l, layer in enumerate(['V1','V2','V4','IT']):
            if not layers or layer in layers:
                activations[layer] = torch.empty(0)
                model[l].register_forward_hook(get_activation(layer))

    elif M.model_name.startswith('resnet'):

        model.relu.register_forward_hook(get_activation('relu'))
        model.layer1.register_forward_hook(get_activation('layer1'))
        model.layer2.register_forward_hook(get_activation('layer2'))
        model.layer3.register_forward_hook(get_activation('layer3'))
        model.layer4.register_forward_hook(get_activation('layer4'))
        model.fc.register_forward_hook(get_activation('fc'))


    elif M.model_name in ['inception_v3']:

        model.Conv2d_1a_3x3.register_forward_hook(get_activation('Conv2d_1a_3x3'))
        model.Conv2d_4a_3x3.register_forward_hook(get_activation('Conv2d_4a_3x3'))
        model.Mixed_5b.register_forward_hook(get_activation('Mixed_5b'))
        model.Mixed_5c.register_forward_hook(get_activation('Mixed_5c'))
        model.Mixed_5d.register_forward_hook(get_activation('Mixed_5d'))
        model.Mixed_6a.register_forward_hook(get_activation('Mixed_6a'))
        model.Mixed_6b.register_forward_hook(get_activation('Mixed_6b'))
        model.Mixed_6c.register_forward_hook(get_activation('Mixed_6c'))
        model.Mixed_6d.register_forward_hook(get_activation('Mixed_6d'))
        model.Mixed_6e.register_forward_hook(get_activation('Mixed_6e'))
        model.Mixed_7a.register_forward_hook(get_activation('Mixed_7a'))
        model.Mixed_7b.register_forward_hook(get_activation('Mixed_7b'))
        model.Mixed_7c.register_forward_hook(get_activation('Mixed_7c'))
        model.fc.register_forward_hook(get_activation('fc'))

    if not layers or 'output' in layers:
        activations['output'] = torch.empty(0)

    #if T.nGPUs > 1: # TODO: fix this code to allow for data parallelism
    #    model = nn.DataParallel(model)
    # loop through batches, collecting activations

    for batch, inputs in enumerate(loader):
            
        inputs = inputs.to(device)
        outputs = model(inputs)

        # inputs and outputs need to be manually appended
        if not layers or 'input' in layers:
            activations['input'] = torch.concat([activations['input'], inputs.detach().cpu()], axis=0)
        if not layers or 'output' in layers:
            if isinstance(outputs, list):
                activations['output'] = torch.concat([activations['output'], outputs[1].detach().cpu()], axis=0)
            else:
                activations['output'] = torch.concat([activations['output'], outputs.detach().cpu()], axis=0)


        # save some input images with class estimates
        if batch == 0 and save_input_samples:
            if 'classification' in M.params_path:
                from utils import response
                if M.out_channels > 1:
                    outputs = outputs[:,:,0]
                responses = response(outputs, 'ILSVRC2012')
            else:
                responses=None
            from utils import save_image_custom
            if op.isdir(sample_input_dir):
                shutil.rmtree(sample_input_dir)
            os.makedirs(sample_input_dir, exist_ok=True)
            save_image_custom(inputs.detach().cpu(), T, sample_input_dir, max_images=128, labels=responses)


    return activations
    

if __name__ == '__main__':
    get_activations(M, image_dir, T, layers=None, sampler=None,
                    norm_minibatch=False, save_input_samples=False,
                    sample_input_dir=None, transform=None, shuffle=False)

