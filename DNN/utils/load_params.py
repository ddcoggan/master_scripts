import torch

def load_params(params, dest_object, object_type, modules='all'):

	# if the path to params is submitted (rather than params themselves), load params
	if type(params) == str:
		params = torch.load(params)

	key_config = {'model': ['model', 'state_dict'],
				  'swa_model': ['swa_model'],
				  'optimizer': ['optimizer'],
				  'swa_optimizer': ['swa_optimizer'],
				  'scheduler': ['scheduler'],
				  'swa_scheduler': ['swa_scheduler']}

	for key in key_config[object_type]:
		if key in params:
			source_params = params[key]
			break

	# resolve key errors arising when 'module.' is prefixed to key
	dest_wrapped = list(dest_object.state_dict().keys())[0].startswith('module')
	source_wrapped = list(source_params.keys())[0].startswith('module')
	if source_wrapped == dest_wrapped:
		resolved_params = source_params
	elif dest_wrapped:
		resolved_params = {f'module.{key}': values for key, values in source_params.items()}
	else:
		resolved_params = {key[7:]: values for key, values in source_params.items()}
		if list(resolved_params.keys())[0].startswith('module'):
			resolved_params = {key[7:]: values for key, values in
							   resolved_params.items()}
	# load params
	if modules == 'all':
		dest_object.load_state_dict(resolved_params)
	else:
		for module in modules:

			# create a new state dict with matching keys to destination object
			module_state_dict = {}
			dest_module_state_dict = getattr(dest_object, module).state_dict()
			for key in dest_module_state_dict:
				resolved_key = f'{module}.{key}'
				if resolved_key not in resolved_params:
					for key_r, tensor_r in resolved_params.items():
						if key_r.endswith(f'{module}.{key}') and \
							tensor_r.shape == dest_module_state_dict[key].shape:
							print('found key')
							resolved_key = key_r
							break
				module_state_dict[key] = resolved_params[resolved_key]
			getattr(dest_object, module).load_state_dict(module_state_dict)
	
	return dest_object
