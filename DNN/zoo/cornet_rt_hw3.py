from collections import OrderedDict
import torch
from torch import nn


HASH = '933c001c'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_RT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp, state=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state


class CORnet_RT(nn.Module):
    
    def __init__(self, hw=3, num_cycles=5, return_states=False,
                 return_blocks=None, return_cycles=None):
        super().__init__()

        # parameters
        self.blocks = ['V1', 'V2', 'V4', 'IT', 'decoder']
        self.num_cycles = num_cycles
        self.return_states = return_states
        self.return_blocks = self.blocks if not return_blocks else return_blocks
        self.return_cycles = torch.arange(num_cycles) if not return_cycles \
            else return_cycles

        # architecture
        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_RT(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_RT(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_RT(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(hw)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(hw**2 * 512, 1000))
        ]))

    # version that stores all states and returns all at once 
    def forward(self, inp):

        num_cycles = inp.shape[0] if len(inp.shape) == 5 else self.num_cycles

        # initialize block states
        stored_cycles = num_cycles if self.return_states else 1
        states = {block: {f'cyc{cycle:02}': 0} for cycle in torch.arange(
            stored_cycles) for block in self.blocks}

        for c in range(num_cycles):
            
            # if image, reinput at each cycle; if movie, input frame sequence
            inp_c = inp if len(inp.shape) == 4 else inp[c]
            
            # if returning states, get previous and current cycles
            if self.return_states:
                prv_cyc, cur_cyc = f'cyc{c-1:02}', f'cyc{c:02}'
            # otherwise, only current cycle is stored
            else:   
                prv_cyc, cur_cyc = f'cyc00', f'cyc00'
            
            for b, block in enumerate(self.blocks[:-1]):

                prv_block = ([None] + self.blocks)[b]

                # get feedforward inputs from inp or prev block, current cycle
                f = inp_c if block == 'V1' else states[prv_block][cur_cyc]

                # get lateral inputs from current block, previous cycle
                l = states[block][prv_cyc] if c > 0 else None

                # forward pass
                outputs = getattr(self, block)(f, l)

                # store outputs
                states[block][cur_cyc] = outputs[1]
            
            # store decoder states for each cycle
            if self.return_states and 'decoder' in self.return_blocks:
                states['decoder'][cur_cyc] = self.decoder(outputs[0])

        if self.return_states:
            # return only requested cycles and blocks
            states = {block: {f'cyc{cycle:02}': states[block][f'cyc{cycle:02}']
                for cycle in self.return_cycles} 
                for block in self.return_blocks}
                
            return states
        else:
            # delete states to free memory and return decoder outputs
            del states
            return self.decoder(outputs[0])

    """
    # version that iteratively yields outputs 
    def forward(self, inp):

        cycles = inp.shape[0] if len(inp.shape) == 5 else self.cycles

        # initialize block states
        blocks = ['V1', 'V2', 'V4', 'IT']
        states = {block: [0] for block in blocks}
        if self.return_states:
            states['decoder'] = None

        for c in range(cycles):

            # if image, reinput at each cycle; if movie, input frame sequence
            inp_c = inp if len(inp.shape) == 4 else inp[c]

            for blk in blocks:
                prv_blk = blocks[
                    blocks.index(blk) - 1] if blk != 'V1' else None

                # get feedforward inputs from inp or prev block, current cycle
                f = inp_c if blk == 'V1' else states[prv_blk]

                # get lateral inputs from current block, previous cycle
                l = states[blk] if c > 0 else None

                # forward pass
                inputs = [f, l]
                outputs = getattr(self, blk)(*inputs)

                # store outputs
                states[blk] = outputs[1]

            # return states for each cycle
            if self.return_states:
                states['decoder'] = self.decoder(outputs[0])
                yield c, states

        # if not self.return_states:
        del states
        return self.decoder(outputs[0])
    """

if __name__ == "__main__":

    model = CORnet_RT(return_states=True)
    params = torch.load('/mnt/HDD2_16TB/projects/p022_occlusion/in_silico/models/cornet_rt_hw3/transform-contrastive/params/best.pt')
    model.load_state_dict(params['model'])
    inputs = torch.rand([5, 3, 224, 224])
    for cycle in model(inputs):
        print(cycle['V1']['f'].shape)

