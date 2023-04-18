import torch
from torchinfo import summary
import math

def calculate_batch_size(model, t, device):

    mod_sum = str(summary(model, input_size=(1,3, 224, 224), device=device)).split('\n')[-4:-2]
    imageMem = float(mod_sum[0].split(' ')[-1])
    imageMemUnit = mod_sum[0].split(' ')[-2][1:3]
    if imageMemUnit == 'KB':
        imageMem *= 1e3
    if imageMemUnit == 'MB':
        imageMem *= 1e6
    if imageMemUnit == 'GB':
        imageMem *= 1e9
    GPUmem = torch.cuda.mem_get_info(device)[0] * t.nGPUs # model is already on device, so get remaining memory
    max_batch = GPUmem / imageMem
    t.batch_size = pow(2, int(math.log(max_batch, 2))) # largest power of 2 below max_batch
    if hasattr(t, 'learning') and 'contrastive' in t.learning:
        t.batch_size /= 2
    t.batch_size = int(t.batch_size)

    return t
