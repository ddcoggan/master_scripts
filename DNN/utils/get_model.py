import os
import sys
import torch.nn as nn
import math
sys.path.append(f'{os.path.expanduser("~")}/david/masterScripts/DNN')
import zoo

def get_model(M):

	if M.model_name.endswith('predify'):
		base_model = M.model_name[:-8]
	else:
		base_model = M.model_name

	model = getattr(zoo, base_model)
		
	
	if base_model in ['cornet_st', 'cornet_flab', 'cornet_s_custom']:
		model = model(M)

		# random initialization for cornet st/ s_custom
		for mod in model.modules():
			if isinstance(mod, nn.Conv2d):
				n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
				mod.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(mod, nn.Linear):
				n = mod.in_features * mod.out_features
				mod.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(mod, nn.BatchNorm2d):
				mod.weight.data.fill_(1)
				mod.bias.data.zero_()
		
		if M.model_name.endswith('predify'):
			if not hasattr(M, 'keep_weights'):
				M.keep_weights = True
			from zoo.cornet_s_custom_predify import cornet_s_custom_predifySeparateHP as cornet_s_custom_predify
			model = cornet_s_custom_predify(model, build_graph=True)#, random_init=True-M.keep_weights)
	else:
		if not hasattr(M, 'pretrained') or M.pretrained is False:
			model = model()
		else:	
			model = model(pretrained=M.pretrained)

	return model
