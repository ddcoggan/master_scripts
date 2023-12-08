import os
import sys
import torch.nn as nn
import math
import zoo

def get_model(model_name, **kwargs):

	try:
		model = getattr(zoo, model_name)(**kwargs)
	except:
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
