import os
import sys
from types import SimpleNamespace
import torch.nn as nn
sys.path.append(f'{os.path.expanduser("~")}/david/masterScripts/DNN')

def change_output_size(model=None, m=None, output_size=None):

	if m.model_name in ['alexnet', 'vgg19']:
		in_features = model.classifier[-1].in_features
		model.classifier.add_module(name=str(len(model.classifier) - 1), module=nn.Linear(in_features, output_size, True))
	elif m.model_name.startswith('cornet_s'):
		in_features = model.decoder[-2].in_features
		model.decoder.add_module(name='linear', module=nn.Linear(in_features, output_size, True))
	elif m.model_name in ['inception_v3'] or m.modelName.startswith('resnet'):
		in_features = model.fc.in_features
		model.add_module(name='fc', module=nn.Linear(in_features, output_size, True))
		model.aux_logits = False
	elif m.model_name.startswith('PredNet'):
		in_features = model.linear.in_features
		model.add_module(name='linear', module=nn.Linear(in_features, output_size, True))
	
	return(model)
