import numpy as np
import torch

def assign_outputs(outputs, CFG):

    # unpack config
    M, D, T = CFG.M, CFG.D, CFG.T

    outputs_class, outputs_contr = None, None

	# separate outputs by classification/contrastive
    # handling for model with multiple different-sized outputs
    # in list form
    if 'output_avpool' in M.model_name:
        outputs_contr, outputs_class = outputs
        outputs_contr = torch.stack(torch.split(outputs_contr, [targets.shape[0]] * D.num_views, dim=0), dim=1)
        outputs_contr = outputs_contr[:, T.views_contr]
        outputs_class = torch.stack(torch.split(outputs_class, [targets.shape[0]] * D.num_views, dim=0), dim=1)
        outputs_class = outputs_class[:, T.views_class]
        outputs_class = torch.concat(torch.split(outputs_class, [1] * outputs_class.shape[1], dim=1),
                                     dim=0).squeeze()


    # unique handling for model with multiple same-sized outputs
    # in tensor form
    elif M.model_name in ['cornet_s_custom', 'cornet_st'] and \
            M.out_channels == 2:
        outputs_class = outputs[:, :, 0]
        outputs_contr = outputs[:, :, 1]

    # normal handling for model with single output
    else:
        if T.contrastive:
            # unstack recombine along view dimension
            outputs = torch.stack(torch.split(
                outputs, [targets.shape[0]] * D.num_views,
                dim=0), dim=1)
            if hasattr(T, 'views_contr'):
                outputs_contr = outputs[:, T.views_contr]
            else:
                outputs_contr = outputs
            if T.classification:
                outputs_class = outputs[:, T.views_class]
                outputs_class = torch.concat(torch.split(
                    outputs_class, [1] * outputs_class.shape[1],
                    dim=1), dim=0).squeeze()
        elif T.classification:
            outputs_class = outputs

    return outputs_class, outputs_contr
