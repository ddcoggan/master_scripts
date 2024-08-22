import numpy as np
import torch

def get_outputs(outputs, M, D, O, criterion):

    views = O.criteria[criterion]['views']

    # handling for model with multiple different-sized outputs in list form
    if 'output_avpool' in M.architecture:
        if 'SimCLR' in criterion:
            outputs = outputs[0]
            outputs = torch.stack(torch.split(outputs, [targets.shape[0]] * D.num_views, dim=0), dim=1)
            outputs = outputs[:, views]
        elif criterion == 'CrossEntropyLoss':
            outputs = outputs[1]
            outputs = torch.stack(torch.split(outputs, [targets.shape[0]] * D.num_views, dim=0), dim=1)
            outputs = outputs[:, views]
            outputs = torch.concat(torch.split(outputs, [1] * outputs.shape[1], dim=1), dim=0).squeeze()

    # handling for model with multiple same-sized outputs in tensor form
    elif M.architecture in ['cornet_s_custom', 'cornet_st'] and \
            M.out_channels == 2:
        if criterion.startswith('SimCLR'): 
            outputs = outputs[:, :, 1]
        elif criterion == 'CrossEntropyLoss':
            outputs = outputs[:, :, 0]
        
    # handling for model with single output but using SimCLR
    elif 'SimCLR' in criterion:
        # unstack and recombine along view dimension
        outputs = torch.stack(torch.split(
            outputs, outputs.shape[0] // D.num_views,
            dim=0), dim=1)[:, views]    
    
    # handle recurrent models by using last cycle's output
    elif type(outputs) is list:
        outputs = outputs[-1]  

    return outputs
