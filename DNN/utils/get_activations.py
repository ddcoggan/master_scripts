import glob
import torch
import torch.nn as nn
import os
import numpy as np
import torchvision.transforms as transforms
from argparse import Namespace
import sys
from torch.utils.data import DataLoader

sys.path.append(os.path.expanduser("~/david/masterScripts/DNN"))

def get_activations(M, image_dir, T, output_only=False, save_input_samples=False, sample_input_dir=None):

    # hardware
    from utils import configure_hardware
    T, device = configure_hardware(T)

    # model
    if not hasattr(M, 'model'):
        print('loading model...')
        from utils import get_model
        model = get_model(M)
    else:
        model = M.model
    model.to(device)
    #if t.nGPUs > 1: # TODO: fix this code to allow for data parallelism
    #    model = nn.DataParallel(model)
    #print(model)
    model.eval()

    # load params
    if not hasattr(M, 'params_loaded') or M.params_loaded is False:
        print('loading parameters...')
        from utils import load_params
        model = load_params(M.params_path, model=model)

    # calculate optimal batch size
    if not hasattr(T, 'batch_size'):
        from utils import calculate_batch_size
        T = calculate_batch_size(model, T, device)
        print(f'optimal batch size calculated at {T.batch_size}')

    # image transforms
    transform_sequence = [transforms.ToTensor(),
                         transforms.Resize(224),  # resize (smallest edge becomes this length)
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    transform = transforms.Compose(transform_sequence)

    # image loader
    from utils import CustomDataSet
    dataset = CustomDataSet(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=T.batch_size, shuffle=False, num_workers=T.workers, pin_memory=True)

    activations = {}
    if not output_only:

        # store activations in a dict with items for each layer
        activations['input'] = torch.empty(0)

        # define activation hook
        def get_activation(idx):
            def hook(model, input, output):
                if type(output) == torch.Tensor:
                    activations[idx] = torch.concat([activations[idx], output.detach().cpu()])
                else:
                    activations[idx].append(output)

            return hook

        # add activation hook to model

        if M.model_name in ['alexnet','vgg19']:
            from utils import model_layer_labels
            layers = model_layer_labels[M.model_name]
            # set up activations dict
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


        elif M.model_name == 'cornet_st':

            for l, layer in enumerate(['V1', 'V2', 'V4', 'IT']):
                activations[layer] = []
                getattr(model, layer).register_forward_hook(get_activation(layer))

        elif M.model_name.startswith('cornet_s'):

            for l, layer in enumerate(['V1','V2','V4','IT']):
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

    activations['output'] = torch.empty(0)

    # loop through batches, collecting activations
    for batch, inputs in enumerate(loader):
            
        inputs = inputs.to(device)
        outputs = model(inputs)

        # inputs and outputs need to be manually appended
        if not output_only:
            activations['input'] = torch.concat([activations['input'], inputs.detach().cpu()], axis=0)
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
    print('to do')
