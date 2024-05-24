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
import gc

import save_image_custom
import predict
import model_layer_labels
import get_model
import calculate_batch_size
import get_transforms

@torch.no_grad()
def get_activations(model=None, model_name=None, image_dir=None, inputs=None,
                    T=SimpleNamespace(num_workers=2), layers=None, sampler=None,
                    norm_minibatch=False, save_input_samples=False,
                    sample_input_dir=None, transform=None, shuffle=False, 
                    array_type='numpy'):

    # hardware
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # model
    if model is None:
        model = get_model(model_name, **{'pretrained': True})
    model.to(device)

    # set batch norm behaviour
    model.train() if norm_minibatch else model.eval()

    # calculate optimal batch size
    if not hasattr(T, 'batch_size'):
        T = calculate_batch_size(model, T, device)
        print(f'optimal batch size calculated at {T.batch_size}')

    # image transforms
    if transform is None:
        _, transform = get_transforms()

    # image loader
    if image_dir is None:
        dataset = TensorDataset(inputs, transform=transform)
    else:
        from DNN.utils import CustomDataSet
        dataset = CustomDataSet(image_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=T.batch_size, shuffle=shuffle,
        num_workers=T.num_workers, sampler=sampler, pin_memory=True)

    # initialize activations dict with one item for each layer
    layers = [layers] if type(layers) is str else layers
    in_out = [l for l in layers if l in ['input', 'output']]
    activations = {l: torch.Tensor(0) for l in in_out}
    layers_hidden = [l for l in layers if l not in in_out]

    # initialize standard dict item
    activations = {**activations,
                   **{l: [] for l in layers_hidden}}

    # define forward hook
    def get_activation(layer):
        def hook(model, input, output):
            # CORnet RT outputs a list of two identical tensors
            if model_name == 'cornet_rt':
                output = output[0]
            activations[layer].append(output.detach().cpu())
        return hook

    # register forward hook
    for layer in layers_hidden:

        # get module by decomposing layer string
        layer_path = layer.split('.')
        module = model
        for l in layer_path:
            module = module[int(l)] if l.isnumeric() else getattr(module, l)

        # only add hook if not already registered
        if not len(module.__dict__['_forward_hooks']):
            module.register_forward_hook(get_activation(layer))

    """
    # define activation hook
    def get_activation(layer):
        def hook(model, input, output):
            if type(output) == torch.Tensor:
                if len(dataset.all_imgs) > T.batch_size:
                    activations[layer] = torch.concat([activations[layer],
                                                     output.detach().cpu()])
                else:
                    # this doesn't blow up memory as much
                    activations[idx] = output.detach().cpu()
            elif 'cornet_rt' in model_name and type(output) == tuple:
                activations[idx] = torch.concat([activations[idx],
                                                 output[0].detach().cpu()])
            else:
                activations[idx].append(output.detach().cpu())

        return hook
    """

    #if T.nGPUs > 1: # TODO: fix this code to allow for data parallelism
    #    model = nn.DataParallel(model)

    # loop through batches
    for batch, inputs in enumerate(tqdm(loader, unit=f"batch({T.batch_size})")):

        if type(inputs) is list:
            inputs = inputs[0]
        inputs = inputs.to(device)
        outputs = model(inputs)

        # inputs and outputs need to be manually appended
        if 'input' in in_out:
            activations['input'] = torch.concat(
                [activations['input'], inputs.detach().cpu()])
        if 'output' in in_out:
            activations['output'] = torch.concat(
                [activations['output'], outputs.detach().cpu()])

        # save some input images with class estimates
        if batch == 0 and save_input_samples:
            try:
                responses = predict(outputs, 'ILSVRC2012')
            except:
                responses=None
            if op.isdir(sample_input_dir):
                shutil.rmtree(sample_input_dir)
            os.makedirs(sample_input_dir, exist_ok=True)
            save_image_custom(inputs.detach().cpu(), sample_input_dir,
                              max_images=128, labels=responses)
            

    # post processing (dont overwrite 'activations' as this somehow screws
    # with the hook even though it is not used after this point)
    activations_post = {}

    # post processing
    for layer in activations:

        # special handling for vision transformer
        if model_name == 'vit_b_16' and layer in layers_hidden:
            patch_size = int(np.sqrt(activations[layer].size(-2)))
            activations_post[layer] = activations[layer][:, 1:, :].reshape(
                activations[layer].size(0), patch_size, patch_size,
                activations[layer].size(2))

        # for recurrent models, separate outputs by cycle
        elif (model_name.startswith('cognet') or model_name in [
              'cornet_s_custom', 'cornet_rt_hw3']) and layer in layers_hidden:
            if model_name.startswith('cognet'):
                cycles = model.cycles
            elif model_name == 'cornet_rt_hw3':
                cycles = model.times
            elif model_name == 'cornet_s_custom':
                layer_idx = (['V1.output', 'V2.output', 'V4.output',
                              'IT.output'].index(layer))
                cycles = model.M.R[layer_idx]
            activations_post[layer] = {f'cyc{c:02}': torch.concat(
                activations[layer][c::cycles], dim=0) for c in range(cycles)}

        # if responses are in a list, concatenate into a tensor
        elif type(activations[layer]) is list:
            activations_post[layer] = torch.concat(
                activations[layer], dim=0)

        # otherwise, just copy the tensor
        else:
            activations_post[layer] = activations[layer]


    # numpy conversion
    if array_type == 'numpy':
        """
        activations_post_np = {}
        for layer, value in activations_post.items():
            if type(value) is dict:
                activations_post_np[layer] = {key: value.numpy() for key,
                    value in activations_post[layer].items()}
            else:
                activations_post_np[layer] = activations_post[layer].numpy()
        return activations_post_np
        """

        def dict_torch_to_numpy(obj):
            for k, v in obj.items():
                if isinstance(v, dict):
                    dict_torch_to_numpy(v)
                else:
                    obj[k] = v.numpy()
        dict_torch_to_numpy(activations_post)


    return activations_post
    

if __name__ == '__main__':
    get_activations(M, image_dir, T, layers=None, sampler=None,
                    norm_minibatch=False, save_input_samples=False,
                    sample_input_dir=None, transform=None, shuffle=False)

