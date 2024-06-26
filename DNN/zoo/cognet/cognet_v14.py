import math
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
import torch
import torchvision.transforms as transforms
from itertools import product as itp

"""
notes over previous version
backward convtransposes performed for all layers
single convolutional layer combining f, l, b signals
zero tensors for l and b signals are created if they are not passed
no padding to stop edge artifacts
no pooling, 
"""

class Identity(nn.Module):
    def forward(self, x):
        return x

class LGN(nn.Module):
    
    def __init__(self, channels, kernel_size=7, stride=1, input_size=224,
                 num_scales=7, max_scale=2, min_scale=1):

        super().__init__()

        self.ic, self.oc = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_size = input_size
        self.num_scales = num_scales
        self.max_scale = max_scale
        self.min_scale = min_scale

        # feed forward convolutions at various eccentricities
        self.scales = torch.linspace(max_scale, min_scale, num_scales)
        self.xforms = nn.ModuleList([transforms.Resize(
            int(input_size // scale), antialias=True) for scale in self.scales])
        self.resize = transforms.Resize(input_size//stride, antialias=True)
        self.windows = torch.linspace(0, input_size // (2 * stride),
                                      num_scales + 1, dtype=torch.int)[:-1]

        # operations
        self.fb = nn.Conv2d(self.ic * 2, self.ic, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(self.ic, self.oc, kernel_size=kernel_size,
                              stride=stride, padding=3, bias=False)
        self.norm = nn.GroupNorm(16, self.oc)
        self.nonlin = nn.ReLU()

        # allow for retrieval with forward hook
        self.f = Identity()


    def forward(self, inputs):

        f, _, b = inputs

        # combine image with feedback signals
        b = torch.zeros_like(f) if b is None else b
        f = self.fb(torch.concat([f, b], dim=1))

        # feed forward
        for transform, w in zip(self.xforms, self.windows):
            temp = transform(f)  # shrink image depending on eccentricity
            temp = self.conv(temp)  # apply convolution
            temp = self.resize(temp)  # grow back to original size
            if w == 0:
                f_out = temp
            else:
                f_out[..., w:-w, w:-w] = temp[..., w:-w, w:-w]

        # final nonlin
        f = self.nonlin(f_out)

        # allow for retrieval with forward hook
        f = self.f(f)
        l = None
        b = None

        return f, l, b


class CogBlock(nn.Module):

    """
    Generic model block for CogNet
    """

    def __init__(self, channels, kernel_size=3, stride=2, scale=4):
        super().__init__()

        self.prev_channels = pc = channels[0]
        self.in_channels = ic = channels[1]
        self.out_channels = oc = channels[2]
        self.stride = stride
        self.scale = scale
        sc = oc * scale

        # integration convolutions
        self.flb = nn.Conv2d(ic * 3, ic, kernel_size=1, bias=False)

        # feed forward connections
        self.conv_skip = nn.Conv2d(oc, oc, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.GroupNorm(16, oc)

        self.conv1 = nn.Conv2d(oc, sc, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(16, sc)
        self.nonlin1 = nn.ReLU()

        self.conv2 = nn.Conv2d(sc, sc, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2, bias=False)
        self.norm2 = nn.GroupNorm(16, sc)
        self.nonlin2 = nn.ReLU()

        self.conv3 = nn.Conv2d(sc, oc, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(16, oc)
        self.nonlin3 = nn.ReLU()

        # lateral connection
        self.conv_lat = nn.ConvTranspose2d(oc, ic, kernel_size=kernel_size,
            stride=stride, padding=1, output_padding=1, bias=False)
        self.norm_lat = nn.GroupNorm(16, ic)
        self.nonlin_lat = nn.ReLU()

        # backward connection
        if self.prev_channels == 3:
            self.conv_back = nn.ConvTranspose2d(
                oc, pc, kernel_size=7, stride=2, padding=3, output_padding=1,
                bias=False)
        else:
            self.conv_back = nn.ConvTranspose2d(oc, pc, kernel_size=5, stride=4,
                padding=1, output_padding=1, bias=False)
        self.norm_back = nn.GroupNorm(min(16, pc), pc)
        self.nonlin_back = nn.ReLU()
        
        # allow for retrieval with forward hook
        self.f = Identity()
        self.l = Identity()
        self.b = Identity()


    def forward(self, inputs):

        f, l, b = inputs

        # combine with lateral and feed back signals
        l = torch.zeros_like(f) if l is None else l
        b = torch.zeros_like(f) if b is None else b
        f = self.flb(torch.concat([f, l, b], dim=1))

        # skip connection
        skip = self.conv_skip(f)
        skip = self.norm_skip(skip)

        # expansion convolution
        f = self.conv1(f)
        f = self.norm1(f)
        f = self.nonlin1(f)

        # middle convolution
        f = self.conv2(f)
        f = self.norm2(f)
        f = self.nonlin2(f)

        # contraction convolution
        f = self.conv3(f)
        f = self.norm3(f)

        # outgoing lateral connection
        l = self.conv_lat(f)
        l = self.norm_lat(l)
        l = self.nonlin_lat(l)
        
        # outgoing backward connection
        b = self.conv_back(f)
        b = self.norm_back(b)
        b = self.nonlin_back(b)

        # combine with skip, final nonlin
        f += skip
        f = self.nonlin3(f)
        
        # allow for retrieval with forward hook
        #f = self.f(f)
        #l = self.l(l)
        #b = self.b(b)

        return f, l, b


class CogDecoder(nn.Module):

    def __init__(self, in_channels, out_features, head_depth, head_width):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.head_depth = head_depth
        self.avgpool = nn.AdaptiveAvgPool2d(head_width)
        self.flatten = nn.Flatten()
        self.head_sizes = torch.linspace(
            in_channels * (head_width ** 2), 1000, head_depth + 1, dtype=int)
        self.f = Identity()

        # flexibly generate decoder based on head_depth
        for layer in range(head_depth):
            setattr(self, f'linear_{layer + 1}',
                    nn.Linear(self.head_sizes[layer],
                              self.head_sizes[layer + 1]))
            if layer < head_depth - 1:
                setattr(self, f'nonlin_{layer + 1}', nn.ReLU())

    def forward(self, inp):

        x = self.avgpool(inp)
        x = self.flatten(x)

        for layer in range(self.head_depth):
            x = getattr(self, f'linear_{layer + 1}')(x)
            if layer < self.head_depth - 1:
                x = getattr(self, f'nonlin_{layer + 1}')(x)

        x = self.f(x)

        return x


class CogNet(nn.Module):

    def __init__(self, cycles=4):
        super().__init__()

        self.cycles = cycles
        chn = [3,64,64,64,64,64]
        hd = 2
        hw = 3

        self.LGN = LGN(channels=chn[:2])
        self.V1 = CogBlock(channels=chn[0:3])
        self.V2 = CogBlock(channels=chn[1:4])
        self.V4 = CogBlock(channels=chn[2:5])
        self.IT = CogBlock(channels=chn[3:6])
        self.decoder = CogDecoder(in_channels=chn[-1], out_features=1000,
                                  head_depth=hd, head_width=hw)
        self.blocks = ['LGN', 'V1', 'V2', 'V4', 'IT', 'decoder']

    def forward(self, inp):

        # adjust cycles if video input is submitted
        cycles = inp.shape[0] if len(inp.shape) == 5 else self.cycles

        # initialize block states
        blocks = self.blocks
        states = {blk: {f'cyc{c:02}': {k: None for k in 'flb'}
                        for c in range(cycles)} for blk in ['input', *blocks]}

        for c in range(cycles):

            cycle = f'cyc{c:02}'
            prev_cycle = f'cyc{c-1:02}'
            
            # if image, reinput at each cycle; if movie, input frame sequence
            states['input'][cycle]['f'] = inp if len(inp.shape) == 4 else inp[c]
            
            for cur_blk in blocks[:-1]:

                # get names of previous and next blocks
                prv_blk = ['input', *blocks][blocks.index(cur_blk)]
                nxt_blk = [*blocks, None][blocks.index(cur_blk) + 1]

                # collate inputs
                f = states[prv_blk][cycle]['f']
                l = states[cur_blk][prev_cycle]['l'] if c else None
                b = states[nxt_blk][prev_cycle]['b'] if (
                        c and cur_blk != 'IT') else None
                inputs = [f, l, b]

                # forward pass
                outputs = getattr(self, cur_blk)(inputs)

                # store outputs
                states[cur_blk][cycle] = {k: v for k, v in zip('flb', outputs)}
            
            states['decoder'][cycle] = self.decoder(states['IT'][cycle]['f'])

        return states['decoder'][cycle]


# plot window borders with each scale a different color (rainbow)
def plot_windows(LGN, outdir):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    os.makedirs(outdir, exist_ok=True)

    # input windows
    fig, ax = plt.subplots()
    input_size = LGN.input_size
    ax.imshow(torch.zeros(3, input_size, input_size).permute(1, 2, 0))
    colors = plt.cm.rainbow(torch.linspace(0, 1, LGN.num_scales))
    for w, win in enumerate(LGN.windows):
        winsize = LGN.input_size - (2*win)
        rect = patches.Rectangle((win, win), winsize, winsize,
                                 linewidth=1, facecolor=colors[w])
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top','left','bottom','right']].set_visible(False)
    plt.savefig(op.join(outdir, 'input_windows.png'))
    plt.close()

if __name__ == "__main__":

    version = 'v14'
    import sys
    from PIL import Image
    import os.path as op
    import os
    import glob
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    model = CogNet(cycles=10)
    plot_windows(model.LGN, op.expanduser(
                 f'~/david/models/cognet_v14/xform-cont/window_plots'))

    sys.path.append(op.expanduser('~/david/master_scripts/image'))
    from image_processing import tile

    sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
    from utils import plot_conv_filters

    model_dir = op.expanduser(f'~/david/projects/p020_activeVision/models/'
                              f'cognet_v10/{version}')
    params_dir = f'{model_dir}/params'
    params_paths = [f'{params_dir}/012.pt', f'{params_dir}/025.pt']

    for params_path in params_paths:
        epoch = op.basename(params_path)[:-3]

        plot_conv_filters('module.LGN.conv.weight', params_path,
                          f'{op.dirname(op.dirname(params_path))}/'
                          f'kernel_plots/epoch-{epoch}_f.png')
        plot_conv_filters('module.V1.conv_back2.weight', params_path,
                          f'{op.dirname(op.dirname(params_path))}/'
                          f'kernel_plots/epoch-{epoch}_l.png')

        feature_maps_dir = (f'{model_dir}/feature_maps/epoch-{epoch}')
        os.makedirs(feature_maps_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
        ])

        model = nn.DataParallel(CogNet(cycles=10, return_states=True).cuda())
        params = torch.load(params_path)
        model.load_state_dict(params['model'])
        batch_size = 4
        data = ImageFolder(f'~/Datasets/ILSVRC2012/val', transform=transform)
        loader = DataLoader(data, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        model = model.module.cpu()
        for batch, (inputs, targets) in enumerate(loader):
            if batch == 0:
                #inputs.cuda()
                states = model(inputs)
                image_paths = {'LGN-f': [], 'V1-b': [], 'sum': []}

                for i in range(batch_size):
                    for cycle in states:
                        images = []
                        for layer, conn in zip(['LGN', 'V1'], ['f', 'b']):
                            image = states[layer][cycle][conn][i].detach().cpu().squeeze()
                            image_array = np.array(image.permute(1, 2, 0))
                            images.append(image_array.copy())
                            central_min = image_array[5:-5,5:-5,:].min()
                            central_max = image_array[5:-5, 5:-5, :].max()
                            image_clip = np.clip(image_array, central_min,
                                                 central_max) - central_min
                            image_scaled = image_clip * (255.0 / central_max)
                            image_PIL = Image.fromarray(image_scaled.astype(np.uint8))
                            outpath = (f'{feature_maps_dir}/{layer}-'
                                       f'{conn}_im{i}_cyc{cycle}.png')
                            image_PIL.save(outpath)
                            image_paths[f'{layer}-{conn}'].append(outpath)

                for maptype, paths in image_paths.items():
                    outpath = f'{feature_maps_dir}/{maptype}_tiled.png'
                    tile(paths, outpath, num_cols=10, base_gap=0,
                         colgap=1, colgapfreq=1,
                         rowgap=8, rowgapfreq=1)
                break

