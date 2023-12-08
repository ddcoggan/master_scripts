import torch.optim as optim
from torch import nn

def get_criteria(T, device):

    criteria = {}
    if T.classification:
        criteria['class'] = nn.CrossEntropyLoss().to(device)
    if T.contrastive:
        from utils import ContrastiveLoss
        criteria['contr'] = ContrastiveLoss().to(device)

    return criteria





