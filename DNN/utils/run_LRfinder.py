"""
This script determines basic training parameters prior to calling the training script.
Specifically, it calculates the maximum batch size and reasonable learning rate, and
resolves conflicts when a training regime is resumed with new settings
"""

import os
import os.path as op
import glob
import sys
import datetime
import numpy as np
import pickle as pkl
import shutil
from types import SimpleNamespace
import pandas as pd
from ignite.handlers import FastaiLRFinder
from ignite.engine import create_supervised_trainer
from .utils import get_loaders, get_optimizer, get_criteria

def find_lr(model, device, D, T):
    
    print('Calculating initial learning rate using LRfinder...')
    
    T.learning_rate = 0.1
    train_loader, _ = get_loaders(D, T, 4)
    lr_finder = FastaiLRFinder()
    optimizer = get_optimizer(model, T)
    criterion = list(get_criteria(T, device).values())[0]
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device)
    model.to(device)
    to_save = {"model": model, "optimizer": optimizer}
    with lr_finder.attach(trainer, to_save=to_save) as finder:
        finder.run(train_loader)
        lr = lr_finder.suggestion()
    
    print(f'Initial learning rate set to {lr:.08}')
    
    return lr

