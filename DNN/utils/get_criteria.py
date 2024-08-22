from torch import nn
import numpy as np

def get_criteria(criteria_names, device):

    criteria = {}
    if type(criteria_names) is str:
        criteria_names = [criteria_names]
    for criterion in criteria_names:
        if hasattr(nn, criterion):
            criteria[criterion] = dict(
                func=getattr(nn, criterion)().to(device), acc1=None,
                acc5=None, sched_metric='acc1', sched_compare=np.greater)
        elif 'SimCLRLoss' in criterion:  # covers both self-supervised and supervised 
            from utils import SimCLRLoss
            criteria[criterion] = dict(func=SimCLRLoss().to(device),
                                sched_metric='SimCLRLoss', sched_compare=np.less)
        criteria[criterion][criterion] = None

    return criteria





