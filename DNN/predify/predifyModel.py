# takes pre-existing model and predify config file and makes module for predifying model that can be called by getModel.py
# config file must be located in predDir already
import os
import sys
from predify import predify
sys.path.append(f'{os.path.expanduser("~")}/david/masterScripts/DNN')
from getModel import getModel

modelName = 'cornet_s_custom'
predDir = f'{os.path.expanduser("~")}/david/masterScripts/DNN/predify/{modelName}'
os.chdir(predDir)

# get model
model = getModel(**{'modelName': modelName, 'times': (1,2,4,2),'RF': (3,3,3,3), 'pretrained': False}).module

# create script to predify the model
predify(model, f'{predDir}/{modelName}_predify.toml')

'''
# import the script
from cornet_s_custom_predify import cornet_s_custom_predifySeparateHP

# predify the model
modelPred = cornet_s_custom_predifySeparateHP(model, build_graph=True)

config = {'modelParams': {'pretrained': False,
                          'outDir': 'cornet_s_custom_predify/unoccluded',
                          'modelName': 'cornet_s_custom_predify',
                          'lastEpoch': None},
          'datasetParams': {'dataset': 'imagenet1000',
                            'datasetPath': f'{os.path.expanduser("~")}/Datasets/imagenet1000',
                            'occluder': 'unoccluded',
                            'propOccluded': .8,
                            'colours': [(0,0,0),(127,127,127),(255,255,255)],
                            'invert': False},
          'trainingParams': {'learningRate': .01,
                             'optimizerName': 'SGD',
                             'nEpochs': 25,
                             'skipZeroth': True,
                             'workers': 8,
                             'nGPUs': 1,
                             'GPUids': 1,
                             'batchSize': 32}}

printTheseParams = ['modelName','times','RF','dataset','lastEpoch']
config['printOut'] = {}
for param in printTheseParams:
    for set in config:
        if param in config[set]:
            config['printOut'][param] = config[set][param]

if config['trainingParams']['nGPUs'] == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config['trainingParams']['GPUids']}"

from train import train
import warnings
warnings.simplefilter("ignore")

train(modelPred, **config)
'''