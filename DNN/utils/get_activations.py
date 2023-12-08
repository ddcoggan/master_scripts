import glob
import torch
import torch.nn as nn
import os
import os.path as op
import numpy as np
import torchvision.transforms as transforms
from argparse import Namespace
import sys
from torch.utils.data import DataLoader, TensorDataset
import shutil
from types import SimpleNamespace
import itertools
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/david/master_scripts/DNN"))
from utils import save_image_custom, predict, model_layer_labels

def get_activations(M, image_dir=None, inputs=None,
                    T=SimpleNamespace(num_workers=2), layers=None, sampler=None,
                    norm_minibatch=False, save_input_samples=False,
                    sample_input_dir=None, transform=None, shuffle=False, 
                    verbose=False, dtype='numpy'):

    torch.no_grad()

    # hardware
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')

    # model
    if not hasattr(M, 'model'):
        print('Loading model...')
        from utils import get_model
        model = get_model(M.model_name, **{'M': M})
    else:
        model = M.model
    model.to(device)

    if verbose: print(model)


    # load params
    if (not hasattr(M, 'params_loaded') or M.params_loaded is False) and (not hasattr(M, 'pretrained') or M.pretrained is False):
        print('Loading parameters...')
        from utils import load_params
        model = load_params(M.params_path, model, 'model')

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
    if image_dir is None:
        dataset = TensorDataset(inputs, transform=transform)
    else:
        from utils import CustomDataSet
        dataset = CustomDataSet(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=shuffle, num_workers=T.num_workers, sampler=sampler, pin_memory=True)

    
    # define activation hook
    def get_activation(idx):
        def hook(model, input, output):
            if type(output) == torch.Tensor:
                if len(dataset.all_imgs) > T.batch_size:
                    activations[idx] = torch.concat([activations[idx], 
                                                     output.detach().cpu()])
                else:
                    # this doesnt blow up memory as much
                    activations[idx] = output.detach().cpu()
            elif 'cornet_rt' in M.model_name and type(output) == tuple:
                activations[idx] = torch.concat([activations[idx], output[0].detach().cpu()])
            else:
                activations[idx].append(output.detach().cpu())
        return hook

    # store activations in a dict with items for each layer
    activations = {}

    if not layers or 'input' in layers:
        activations['input'] = torch.Tensor(0)


    # add activation hook to model
    if M.model_name in ['alexnet', 'vgg19']:
        for layer in layers:
            activations[layer] = []
            if layer in model_layer_labels[M.model_name]:
                l = model_layer_labels[M.model_name].index(layer)
                try:
                    model.features[l].register_forward_hook(get_activation(layers[layer_counter]))
                    layer_counter += 1
                except:
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

    elif M.model_name == 'cornet_rt_hw3' and model.return_states:
        activations = {**activations, **{l: {f'cyc{c:02}': torch.Tensor(0)
            for c in range(model.num_cycles)} for l in layers}}
        
    elif M.model_name == 'cornet_st' or M.model_name.startswith('cornet_rt'):
        for l, layer in enumerate(['V1', 'V2', 'V4', 'IT']):
            if not layers or layer in layers:
                activations[layer] = torch.Tensor(0)
                getattr(model, layer).register_forward_hook(get_activation(layer))

    elif M.model_name == 'cornet_s_custom' and M.return_states:
        for l, layer in enumerate(['V1', 'V2', 'V4', 'IT']):
            if not layers or layer in layers:
                activations[layer] = {f'cyc{c:02}': torch.Tensor(0)
                                      for c in range(M.R[l])}
        if not layers or 'decoder' in layers:
            activations['decoder'] = {'cyc00': torch.Tensor(0)}


    elif M.model_name.startswith('cornet'):

        for l, layer in enumerate(['V1','V2','V4','IT']):
            if not layers or layer in layers:
                activations[layer] = torch.Tensor(0)
                model[l].register_forward_hook(get_activation(layer))

    elif M.model_name.startswith('resnet'):

        model.relu.register_forward_hook(get_activation('relu'))
        model.layer1.register_forward_hook(get_activation('layer1'))
        model.layer2.register_forward_hook(get_activation('layer2'))
        model.layer3.register_forward_hook(get_activation('layer3'))
        model.layer4.register_forward_hook(get_activation('layer4'))
        model.fc.register_forward_hook(get_activation('fc'))

    elif M.model_name == 'VOneNet_resnet50':
        activations['vone_block'] = []
        model.vone_block.register_forward_hook(get_activation('vone_block'))


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
        activations['output'] = []

    #if T.nGPUs > 1: # TODO: fix this code to allow for data parallelism
    #    model = nn.DataParallel(model)
    # loop through batches, collecting activations

    # loop through batches
    with tqdm(loader, unit=f"batch({T.batch_size})") as tepoch:
        for batch, inputs in enumerate(tepoch):
    

            if type(inputs) is list:
                inputs = inputs[0]
            inputs = inputs.to(device)
            outputs = model(inputs)

            # special handling for cornet_rt_hw3
            if M.model_name in ['cornet_rt_hw3', 'cornet_s_custom'] and \
                    M.return_states:
                for layer in layers:
                    for cycle, states in outputs[layer].items():
                        activations[layer][cycle] = torch.cat(
                            [activations[layer][cycle], 
                             states.detach().cpu()], dim=0)

            # inputs and outputs need to be manually appended
            if not layers or 'input' in layers:
                activations['input'].append(inputs.detach().cpu())
            if not layers or 'output' in layers:
                activations['output'].append(outputs.detach().cpu())

            # save some input images with class estimates
            if batch == 0 and save_input_samples:
                if hasattr(M, 'out_channels') and M.out_channels > 1:
                    outputs = outputs[:,:,0]
                try:
                    responses = predict(outputs, 'ILSVRC2012')
                except:
                    responses=None
                if op.isdir(sample_input_dir):
                    shutil.rmtree(sample_input_dir)
                os.makedirs(sample_input_dir, exist_ok=True)
                save_image_custom(inputs.detach().cpu(), sample_input_dir, max_images=128, labels=responses)

    # combine batches of tensors into single numpy array
    if dtype == 'numpy':
        activations_np = {}
        if M.model_name in ['cornet_rt_hw3', 'cornet_s_custom'] and \
                                             M.return_states:
            for layer in activations:
                activations_np[layer] = {}
                for cycle in activations[layer]:
                    activations_np[layer][cycle] = activations[layer][cycle].numpy()
        else:
            for layer in activations:
                if not isinstance(activations[layer], list):
                    activations_np[layer] = activations[layer].numpy()
                else:
                    activations_np[layer] = np.concatenate(
                        [x.numpy() for x in activations[layer]], axis=0)
        return activations_np
    
    else:
        return activations
    

if __name__ == '__main__':
    get_activations(M, image_dir, T, layers=None, sampler=None,
                    norm_minibatch=False, save_input_samples=False,
                    sample_input_dir=None, transform=None, shuffle=False)

