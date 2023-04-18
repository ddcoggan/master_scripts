import torch

def load_params(params, model=None, optimizer=None):

	# if the path to params is submitted (rather than params themselves), load params
	if type(params) == str:
		params = torch.load(params)

	if model:
		try:
			model_params = params['model']
		except:
			model_params = params['state_dict']
		params_loaded = False
		try:  # try to load model as is
			model.load_state_dict(model_params)
			params_loaded = True
		except:

			new_params = model_params.copy()
			while list(new_params.keys())[0].startswith('module.'):

				# remove 'module.' prefix from each key
				new_params = {}
				for key in model_params:
					new_key = key[7:]
					new_params[new_key] = model_params[key]

				# try to load params
				try:
					model.load_state_dict(new_params)
					params_loaded = True
				except:
					continue

			if not params_loaded:

				# add 'module.' prefix to each key
				new_params = {}
				for key in model_params:
					new_params[f'module.{key}'] = model_params[key]

				# try to load params
				try:
					model.load_state_dict(new_params)
				except:
					Exception('Model parameters failed to load.')
		
	if optimizer:
		optimizer.load_state_dict(params['optimizer'])

	if model and optimizer:
		return model, optimizer
	elif model:
		return model
	if optimizer:
		return optimizer
