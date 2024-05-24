import os
import os.path as op
import sys
import torch.nn as nn
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import zoo
from torchvision import models

def get_model(model_name, kwargs):

	if hasattr(models, model_name):
		try:
			model = getattr(models, model_name)(**kwargs)
		except:
			UserWarning('kwargs not accepted for this model, ignoring...')
			model = getattr(models, model_name)()
	else:
		try:
			model = getattr(zoo, model_name)(**kwargs)
		except:
			UserWarning('kwargs not accepted for this model, ignoring...')
			model = getattr(zoo, model_name)()

	# random initialization
	for mod in model.modules():
		if isinstance(mod, nn.Conv2d):
			n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
			mod.weight.data.normal_(0, math.sqrt(2. / n))
		elif isinstance(mod, nn.Linear):
			n = mod.in_features * mod.out_features
			mod.weight.data.normal_(0, math.sqrt(2. / n))
			mod.bias.data.zero_()
		elif isinstance(mod, nn.BatchNorm2d):
			mod.weight.data.fill_(1)
			mod.bias.data.zero_()

	return model
