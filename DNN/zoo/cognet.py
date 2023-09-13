import math
from collections import OrderedDict
from torch import nn, stack
from argparse import Namespace
import torch
import numpy as np

class CogV1(nn.Module):
    
    """
    CogNet V1 block
    This is designed to model simple and complex cells in V1.
    In order to be treated like a CogBlock, it accepts maxpooling indices in 
    the forward pass, and returns feedback signals in the backward pass, 
    but these are ignored.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        # feed forward connections
        self.simple = nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.nonlin = nn.ReLU()
        self.complex = nn.MaxPool2d(2,2, return_indices=True)

        # lateral connections
        self.conv_lateral = nn.Conv2d(out_channels, in_channels, kernel_size=3,
                                      padding=1, bias=False)
        self.norm_lateral = nn.BatchNorm2d(in_channels)
        self.nonlin_lateral = nn.ReLU()


    def forward(self, f, l, b, i=None):

        # combine image with lateral and feedback signals
        f = f + l + b

        # feed forward
        f = self.simple(f)
        f = self.norm(f)

        # outgoing lateral connection
        l = self.conv_lateral(f)
        l = self.norm_lateral(l)
        l = self.nonlin_lateral(l)

        # final nonlin and maxpool
        f = self.nonlin(f)
        f, i = self.complex(f)
        
        b = None
        
        return f, l, b, i


class CogBlock(nn.Module):

    """
    Generic model block for CogNet
    """

    def __init__(self, channels, kernel_size, stride=1, scale=4):
        super().__init__()

        self.prev_channels = pc = channels[0]
        self.in_channels = ic = channels[1]
        self.out_channels = oc = channels[2]
        self.stride = stride
        self.scale = scale
        sc = oc * scale

        # feed forward connections
        self.conv_input = nn.Conv2d(ic, oc, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size//2)
        self.conv_skip = nn.Conv2d(oc, oc, kernel_size=1, stride=1, bias=False)
        self.norm_skip = nn.BatchNorm2d(oc)

        self.conv1 = nn.Conv2d(oc, sc, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(sc)
        self.nonlin1 = nn.ReLU()

        self.conv2 = nn.Conv2d(sc, sc,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size//2, bias=False)
        self.norm2 = nn.BatchNorm2d(sc)
        self.nonlin2 = nn.ReLU()

        self.conv3 = nn.Conv2d(sc, oc, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(oc)
        self.nonlin3 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2,
                                    return_indices=True)

        # lateral connection
        self.conv_lateral = nn.Conv2d(oc, ic, kernel_size=3, padding=1,
                                      bias=False)
        self.norm_lateral = nn.BatchNorm2d(ic)
        self.nonlin_lateral = nn.ReLU()

        # backward connection
        self.unpool_back = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_back = nn.ConvTranspose2d(oc, pc, kernel_size=3, padding=1,
                                            bias=False)
        self.norm_back = nn.BatchNorm2d(pc)
        self.nonlin_back = nn.ReLU()


    def forward(self, f, l, b, i):

        # input convolution
        f = self.conv_input(f)

        # combine with lateral and feed back signals
        f = f + l + b

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

        # combine with skip
        f += skip

        # outgoing lateral connection
        l = self.conv_lateral(f)
        l = self.norm_lateral(l)
        l = self.nonlin_lateral(l)

        # outgoing backward connection
        b = self.unpool_back(f, i)
        b = self.conv_back(b)
        b = self.norm_back(b)
        b = self.nonlin_back(b)

        # final nonlin and avgpool
        f = self.nonlin3(f)
        f, i = self.maxpool(f)

        return f, l, b, i


class CogDecoder(nn.Module):

    def __init__(self, in_channels, out_features, head_depth, head_width):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.head_depth = head_depth
        self.avgpool = nn.AdaptiveAvgPool2d(head_width)
        self.flatten = nn.Flatten()
        self.head_sizes = np.linspace(in_channels * (head_width ** 2), 1000,
                                 head_depth + 1, dtype=int)

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

        return x


class CogNet(nn.Module):

    def __init__(self, cycles=5, return_states=False):
        super().__init__()

        self.cycles = cycles
        self.return_states = return_states

        chn = [3,64,64,64,64,64]
        hd = 2
        hw = 3

        self.V1 = CogV1(in_channels=chn[0], out_channels=chn[1], kernel_size=7,
                        stride=1)
        self.V2 = CogBlock(channels=chn[0:3], kernel_size=3, stride=1, scale=4)
        self.V3 = CogBlock(channels=chn[1:4], kernel_size=3, stride=1, scale=4)
        self.V4 = CogBlock(channels=chn[2:5], kernel_size=3, stride=1, scale=4)
        self.IT = CogBlock(channels=chn[3:6], kernel_size=3, stride=1, scale=4)
        self.decoder = CogDecoder(in_channels=chn[-1], out_features=1000,
                                  head_depth=hd, head_width=hw)

    def forward(self, inp):

        # adjust cycles if video input is submitted
        cycles = inp.shape[0] if len(inp.shape) == 5 else self.cycles

        # initialize block states
        blocks = ['V1', 'V2', 'V3', 'V4', 'IT']
        stored_cycles = cycles if self.return_states else 1
        states = {f'cyc{cycle:02}': {block: {
            'f': 0, 'l': 0, 'b': 0, 'i': 0
        } for block in blocks} for cycle in np.arange(stored_cycles)}
        if self.return_states:
            for cycle in np.arange(stored_cycles):
                states[f'cyc{cycle:02}']['decoder'] = None

        for c in range(cycles):
            
            # if image, reinput at each cycle; if movie, input frame sequence
            inp_c = inp if len(inp.shape) == 4 else inp[c]
            
            # if returning states, get previous and current cycles
            if self.return_states:
                prv_cyc, cur_cyc = f'cyc{c-1:02}', f'cyc{c:02}'
            # otherwise, only current cycle is stored
            else:   
                prv_cyc, cur_cyc = f'cyc00', f'cyc00'
            
            for blk in blocks:
                
                prv_blk = blocks[blocks.index(blk) - 1] if blk != 'V1' else None
                nxt_blk = blocks[blocks.index(blk) + 1] if blk != 'IT' else None
                    
                # get feedforward inputs from inp or prev block, current cycle
                f = inp_c if blk == 'V1' else states[cur_cyc][prv_blk]['f']

                # get lateral inputs from current block, previous cycle
                l = states[prv_cyc][blk]['l'] if c > 0 else 0

                # get feedback inputs from next block, previous cycle
                b = 0 if blk == 'IT' or c == 0 else \
                    states[prv_cyc][nxt_blk]['b']

                # get maxpool indices from previous block
                i = states[cur_cyc][prv_blk]['i'] if blk != 'V1' else 0

                # forward pass
                f, l, b, i = getattr(self, blk)(f, l, b, i)

                # store outputs
                states[cur_cyc][blk] = {'f': f, 'l': l, 'b': b, 'i': i}
            
            # store decoder states for each cycle
            if self.return_states:
                states[cur_cyc]['decoder'] = self.decoder(f)

        if self.return_states:
            return states
        else:
            return self.decoder(f)


if __name__ == "__main__":


    import sys
    from PIL import Image
    import os.path as op
    import glob
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    sys.path.append(op.expanduser('~/david/master_scripts/image'))
    from image_processing import tile

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])

    model = nn.DataParallel(CogNet(cycles=10, return_states=True).cuda())
    params = torch.load('/mnt/HDD2_16TB/projects/p022_occlusion/in_silico'
                        '/models/cognet/v1/params/013.pt')
    model.load_state_dict(params['model'])
    batch_size = 8
    data = ImageFolder(f'~/Datasets/ILSVRC2012/val', transform=transform)
    loader = DataLoader(data, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    for batch, (inputs, targets) in enumerate(loader):
        if batch == 0:
            inputs.cuda()
            states = model(inputs)
            for i in range(batch_size):
                for cycle in states:
                    image = states[cycle]['V2']['b'][i].detach().cpu().squeeze()
                    image_array = np.array(image.permute(1, 2, 0))
                    image_pos = image_array - image_array.min()
                    image_scaled = image_pos * (255.0 / image_pos.max())
                    image_PIL = Image.fromarray(image_scaled.astype(np.uint8))
                    outpath = f'/home/tonglab/Desktop/cognet/b_{i}_{cycle}.png'
                    image_PIL.save(outpath)

                    image = states[cycle]['V1']['l'][i].detach().cpu().squeeze()
                    image_array = np.array(image.permute(1, 2, 0))
                    image_pos = image_array - image_array.min()
                    image_scaled = image_pos * (255.0 / image_pos.max())
                    image_PIL = Image.fromarray(image_scaled.astype(np.uint8))
                    outpath = (f'/home/tonglab/Desktop/cognet/l_{i}'
                               f'_{cycle}.png')
                    image_PIL.save(outpath)
            break

    for conn in ['b','l']:
        outpath = f'/home/tonglab/Desktop/cognet/tiled_{conn}.png'
        image_paths = sorted(glob.glob(f'/home/tonglab/Desktop/cognet/'
                                       f'{conn}*.png'))
        tile(image_paths, outpath, num_cols=10, base_gap=0, colgap=0, colgapfreq=0,
             rowgap=8, rowgapfreq=1)
