import pandas as pd
import os.path as op

def response(output, dataset, type='label', afc=None):

    """returns classification guess (based on directory label) for all classes or within alternate forced choice"""
    if dataset == 'ILSVRC2012':
        label_data = pd.read_csv(open(op.expanduser('~/david/datasets/images/ILSVRC2012/labels.csv'), 'r+'))
    if afc:
        output = output[:,afc]
    class_idx = output.argmax(dim=1)
    if afc:
        class_idx = [afc[idx] for idx in class_idx]

    responses = [label_data[type][int(idx)] for idx in class_idx]
    return responses
