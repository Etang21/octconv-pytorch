"""
Implementation of the Octave Convolution module in PyTorch.

OctConv2d is our building block. Takes in x_h, x_l and returns out_h and out_l.

OctConv2dStackable implements some tensor rearrangements to achieve the same functionality
with just one input and one output.

To address:
## gpu
## downsampling

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OctConv2d(nn.Module):
    """
    OctConv2D layer which takes in x_h and x_l and returns out_h and out_l.
    
    Building block layer for OctConv2DStackable.
    """
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in, alpha_out, freq_ratio=2, stride=1, padding=0):
        """
        Initializes an Octave Convolution (OctConv) layer.
        
        Inputs:
        - in_channels: Number of input channels
        - out_channels: Number of output channels
        - kernel_size: Int/tuple representing kernel size (HH, HH)/(HH, WW) of convolution
        - alpha_in: Float/sequence of proportions of incoming channels that are low frequency
            - each alpha_in * in_channels must be an integer
            - sum(alpha_in) must be less than 1
        - alpha_out: Float/sequence of proportions of outgoing channels that are low frequency
            - each alpha_out * out_channels must be an integer
            - sum(alpha_out) must be less than 1
        - freq_ratio: List/sequence of ratios of high frequency to low frequency
        - stride: Int/tuple of stride for convolutional kernel
        - padding: Int/tuple of padding for convolutional kernel
            - good to have 2*padding - kernel_size + stride = 0 for the height and width dimensions, otherwise may have 
              tensor size issues (i.e. fail the o_freqs condition in forward)
        
            
        # TODO: Add other params that nn.Conv2d also supports
        """
        super().__init__()
        
        # basic setup and assertions
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding
        
        if isinstance(alpha_in, float) or isinstance(alpha_in, int): alpha_in = [alpha_in]
        if isinstance(alpha_out, float) or isinstance(alpha_out, int): alpha_out = [alpha_out]
        if isinstance(freq_ratio, float) or isinstance(freq_ratio, int): freq_ratio = [freq_ratio]
        alpha_in = list(alpha_in); alpha_out = list(alpha_out); freq_ratio = list(freq_ratio)
        assert len(alpha_in) == len(freq_ratio), "alpha_in and freq_ratio are not of the same length"
        assert len(alpha_out) == len(freq_ratio), "alpha_out and freq_ratio are not of the same length"
        alpha_in.insert(0, 1 - sum(alpha_in))
        alpha_out.insert(0, 1 - sum(alpha_out))
        freq_ratio.insert(0, 1)
        assert alpha_in[0] > 0, "no fundamental frequency input"
        assert alpha_out[0] > 0, "no fundamental frequency output"
        self.freq_ratio = freq_ratio
        
        # Compute number of high-freq and low-freq channels
        self.in_channels_list = []
        self.out_channels_list = []
        for ii in range(len(freq_ratio)):
            assert float(alpha_in[ii] * in_channels).is_integer(), "Incompatible in_channels, alpha_in"
            assert float(alpha_out[ii] * out_channels).is_integer(), "Incompatible out_channels, alpha_out"
            self.in_channels_list.append(int(alpha_in[ii] * in_channels))
            self.out_channels_list.append(int(alpha_out[ii] * out_channels))        
        
        # Check for low-freq outputs or inputs
        self.has_in_list = [in_channel > 0 for in_channel in self.in_channels_list]
        self.has_out_list = [out_channel > 0 for out_channel in self.out_channels_list]
        
        # Create conv and sampling (upsample + pool) layers
        # self.conv_dict[(i,j)] stores conv weights W^{i -> j}. Analogous notation for self.sampling_dict.
        self.conv_dict = {}
        self.sampling_dict = {}
        self.eps = 0.00001 # in case of upsampling round-off
        for i in range(len(self.in_channels_list)):
            for j in range(len(self.out_channels_list)):
                if not self.has_in_list[i] or not self.has_out_list[j]:
                    self.conv_dict[(i,j)] = None
                else:
                    self.conv_dict[(i,j)] = nn.Conv2d(self.in_channels_list[i], self.out_channels_list[j], kernel_size, stride=stride, padding=padding)
                    if self.freq_ratio[i] > self.freq_ratio[j]:
                        scale_factor = self.freq_ratio[i] / self.freq_ratio[j] + self.eps
                        self.sampling_dict[(i,j)] = nn.Upsample(scale_factor=scale_factor, mode='nearest')
                    elif self.freq_ratio[i] < self.freq_ratio[j]:
                        pool_stride = self.freq_ratio[j] / self.freq_ratio[i]
                        if float(pool_stride).is_integer():
                            self.sampling_dict[(i,j)] = nn.AvgPool2d((int(pool_stride), int(pool_stride)), stride=int(pool_stride))
                        else:
                            self.sampling_dict[(i,j)] = nn.Upsample(scale_factor=1/pool_stride, mode='bilinear') # roughly the same effect
                    else:
                        self.sampling_dict[(i,j)] = None
    
        
    def forward(self, x_freqs):
        """
        Computes a forward pass for the OctConv layer.
        
        Inputs:
        - x_freqs: list of input data for each frequency
            - The channels dimension of the data for each frequency must match the alpha_in provided in __init__
            - For each non-zero low-frequency input, its H' and W' must equal H/fr and W/fr where H and W are the dimensions for 
              the highest-frequency data, and fr is the ratio of highest frequency to low frequency
            - To satisfy condition for o_freqs, good to have H'/W' divisible by stride_H/stride_W respectively, 
              otherwise may have tensor size issues (i.e. fail the o_freqs condition below)
        # - x_h: Input high-frequency data, of shape (N, (1 - alpha_in) * C, H, W)
        # - x_l: Input low-frequency data, of shape(N, alpha_in * C, H / freq_ratio, W / freq_ratio)
            # - If alpha_in = 0, x_l can be anything
        
        Returns a tuple of:
        - o_freqs: list of output data for each frequency
            - For each non-zero low-frequency output, its H' and W' must equal H/fr and W/fr where H and W are the dimensions for
              the highest-frequency data, and fr is the ratio of highest frequency to low frequency
        # - out_h: High-frequency outputs, of shape (N, (1 - alpha_out) * F, H', W')
        # - out_l: Low-frequency outputs, of shape (N, alpha_out * F, H' / freq_ratio, W' / freq_ratio) 
                      or None if alpha_out = 0
        """
        for i in range(1, len(self.in_channels_list)):
            if x_freqs[i] is not None:
                assert x_freqs[i].shape[2]*self.freq_ratio[i] == x_freqs[0].shape[2], "OctConv input width does not match frequency ratio"
                assert x_freqs[i].shape[3]*self.freq_ratio[i] == x_freqs[0].shape[3], "OctConv input height does not match frequency ratio"
        
        o_freqs = []
        for j in range(len(self.out_channels_list)):
            out_j = None
            if self.has_out_list[j]:     
                for i in range(len(self.in_channels_list)):
                    if self.has_in_list[i]:
                        out_ij = self.conv_dict[(i,j)](x_freqs[i])
                        if self.sampling_dict[(i,j)] is not None:
                            out_ij = self.sampling_dict[(i,j)](out_ij)
                        if out_j is None:
                            out_j = out_ij
                        else:
                            out_j += out_ij
            o_freqs.append(out_j)
        
        for j in range(1, len(self.out_channels_list)):
            if o_freqs[j] is not None:
                assert o_freqs[j].shape[2]*self.freq_ratio[j] == o_freqs[0].shape[2], "OctConv output width does not match frequency ratio"
                assert o_freqs[j].shape[3]*self.freq_ratio[j] == o_freqs[0].shape[3], "OctConv output height does not match frequency ratio"
        
        return o_freqs

    def decompose_input(self, input):
        """
        Disassembles Numpy array into x_freqs for consumption by OctConv2d.
        
        Inputs:
        (MODIFIED -- Now takes input in the form x_h, (freq_ratio[1] x freq_ratio[1] square of x_l1's)), etc.
        (x_li is cut into freq_ratio[i]**2 pieces N wise, then stacked as follows, where we're observing from
        (the bottom zero index of each tensor:
        ( 0                1                    2                    ... -- W
        ( freq_ratio[i]    freq_ratio[i] + 1    freq_ratio[i] + 2    ...
        ( 2*freq_ratio[i]  2*freq_ratio[i] + 1  2*freq_ratio[i] + 2  ...
        ( ...
        (  |
        (  H
        
        """
        split_sizes = [self.in_channels_list[0]]
        for i in range(1, len(self.in_channels_list)):
            split_sizes.append(1 + (self.in_channels_list[i] - 1)//(self.freq_ratio[i]**2))
        x_all = torch.split(input, split_sizes, dim=1)
        x_freqs = [x_all[0]]
        for i in range(1, len(x_all)):
            x_l_aggregate = x_all[i]
            if 0 in x_l_aggregate.shape: # if there is no input of given frequency mode
                x_freqs.append(None)
            else:
                freq_r = self.freq_ratio[i]
                x_l_list  = [x_l_slice \
                             for x_l_row_agg in torch.split(x_l_aggregate, x_l_aggregate.shape[2]//freq_r, dim=2) \
                             for x_l_slice in torch.split(x_l_row_agg, x_l_row_agg.shape[3]//freq_r, dim=3)]
                x_l = torch.cat(x_l_list, dim=1)
                # slice off zero-padding
                spillover = self.in_channels_list[i] % (freq_r**2)
                if spillover != 0:
                    x_l = torch.split(x_l, [x_l.shape[1] - (freq_r**2 - spillover), freq_r**2 - spillover], dim=1)[0]
                x_freqs.append(x_l)
        return x_freqs
    
    def compose_output(self, o_freqs):
        """
        Reassembles o_freqs into an output numpy array.
        
        Uses the reverse of the method documented in decompose_input
        """        
        out = o_freqs[0]
        for j in range(1, len(o_freqs)):
            out_l = o_freqs[j]
            if out_l is not None:
                freq_r = self.freq_ratio[j]
                # pad zeros for stacking
                spillover = out_l.shape[1] % (freq_r**2)
                if spillover != 0:
                    out_l = F.pad(out_l, (0, 0, 0, 0, 0, freq_r**2 - spillover), 'constant', 0) 
                    # pytorch documentation warns of nondeterministic behavior in backpropagation when using F.pad??
                out_l_slices = torch.split(out_l, out_l.shape[1]//(freq_r**2), dim=1)
                out_l_aggregate = torch.cat([torch.cat([out_l_slices[ii] for ii in range(jj*freq_r, (jj+1)*freq_r)], dim=3) \
                                             for jj in range(freq_r)], dim=2)
                out = torch.cat((out, out_l_aggregate), dim=1)
        return out
        
        
class OctConv2dStackable(OctConv2d):
    """
    Wrapper for OctConv2d. Takes in one numpy array as input and outputs one as output.
    
    Useful for adding to sequential modules!
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in, alpha_out, freq_ratio=2, stride=1, padding=0):
        """
        Initializes an Octave Convolution (OctConv) layer.
        
        Inputs:
        - in_channels: Number of input channels
        - out_channels: Number of output channels
        - kernel_size: Tuple representing kernel size (HH, WW)
        - alpha_in: Proportion of incoming channels that are low frequency
            - alpha_in * in_channels must be an integer
        - alpha_out: Porportion of outgoing channels that are low frequency
            - alpha_out * out_channels must be an integer
        - stride: Stride of convolutional kernel
        - padding: Padding for convolutional kernel
        - freq_ratio: ratio of high frequency to low frequency
        """
        super().__init__(in_channels, out_channels, kernel_size, 
                         alpha_in, alpha_out, freq_ratio=freq_ratio, stride=stride, padding=padding)
        
    def forward(self, input):
        """
        Computes a forward pass for the OctConv layer.
        
        Inputs:
        (MODIFIED -- Now takes input in the form x_h, (freq_ratio x freq_ratio square of x_l's))
        (x_l is cut into freq_ratio**2 pieces N wise, then stacked as follows, where we're observing from
        (the bottom zero index of each tensor:
        ( 0             1                 2                 ... -- W
        ( freq_ratio    freq_ratio + 1    freq_ratio + 2    ...
        ( 2*freq_ratio  2*freq_ratio + 1  2*freq_ratio + 2  ...
        ( ...
        (  |
        (  H

        Returns:
        - Analogously structured numpy array containing out_h and out_l.
        """
        # Decompose input into x_l and x_h using the food-in-fridge method
        x_freqs = super().decompose_input(input)

        # Apply Octave Convolutions
        o_freqs = super().forward(x_freqs)
        
        # Rebuild the h - freq_ratio x freq_ratio _ l shape tensor
        return super().compose_output(o_freqs)
    
class OctConv2dBN(OctConv2d):
    """
    Implements an OctConv2d layer with batch normalization for each feature map.
    Also uses the tricks from OctConv2dStackable to take one input and one output.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in, alpha_out, freq_ratio=2, stride=1, padding=0):
        """
        Initializes OctConv2d layer with batchnorm for both frequencies.
        Further documentation in OctConv2d class.
        """
        super().__init__(in_channels, out_channels, kernel_size, 
                 alpha_in, alpha_out, freq_ratio=freq_ratio, stride=stride, padding=padding)
        
        if isinstance(alpha_in, float) or isinstance(alpha_in, int): alpha_in = [alpha_in]
        alpha_in = list(alpha_in)
        alpha_in.insert(0, 1 - sum(alpha_in))
        
        self.bn_list = [nn.BatchNorm2d(int(in_channels * alpha_in[i])) for i in range(len(alpha_in))]

    
    def forward(self, input):
        """
        Computes a forward pass for the OctConv layer.
        Applies decomposition -> batchnorm -> octconv -> composition.
        """
        x_freqs = self.decompose_input(input)
        
        for i in range(len(self.bn_list)):
            if self.bn_list[i].num_features > 0:
                x_freqs[i] = self.bn_list[i](x_freqs[i])
            
        o_freqs = super().forward(x_freqs)
        return self.compose_output(o_freqs)
    
        
def flatten(x):
    """
    Flsttens input vector, preserves first dimension
    
    Code taken from Assignment 2, Pytorch notebook
    """
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    """
    Need this too XD
    """
    def forward(self, x):
        return flatten(x)

## TODO: formulate get_stacked_4(BN) to above generalized format
    
def get_stacked_4(alpha, freq_ratio, hidden_channels, C, H, W, D_out):
    """
    Returns stacked 4 layer octConvNet as in modules, using stacked format.

    """
    model = nn.Sequential(
        OctConv2dStackable(C, hidden_channels, (3, 3), 0, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, 0, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(hidden_channels * (H // 4) * (W // 4), D_out)
        )
    return model

def get_stacked_4BN(alpha, freq_ratio, hidden_channels, C, H, W, D_out):
    """
    Returns stacked 4 layer octConvNet with Batchnorm at each OctConv layer.

    """
    model = nn.Sequential(
        OctConv2dBN(C, hidden_channels, (3, 3), 0, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, 0, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(hidden_channels * (H // 4) * (W // 4), D_out)
        )
    return model

def get_SixLayerConvNet():
    """
    Returns vanilla convolutional network with six convolutional layers.
    
    Current implementation does not use Batchnorm. We should add Batchnorm to OctConv layers soon for a fair comparison.
    """
    channels = [3, 32, 32, 32, 32, 32, 32]
    fc_1 = 32
    num_classes = 10
    
    model = nn.Sequential(
        nn.Conv2d(channels[0], channels[1], (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(channels[1]),
        nn.Conv2d(channels[1], channels[2], (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(channels[2]),
        nn.Conv2d(channels[2], channels[3], (2, 2), padding=0, stride=2), # Downsamples
        nn.ReLU(),
        nn.BatchNorm2d(channels[3]),
        nn.Conv2d(channels[3], channels[4], (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(channels[4]),
        nn.Conv2d(channels[4], channels[5], (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(channels[5]),
        nn.Conv2d(channels[5], channels[6], (2, 2), padding=0, stride=2), # Downsamples
        nn.ReLU(),
        Flatten(),
        nn.Linear(channels[6] * 8 * 8, fc_1),
        nn.ReLU(),
        nn.Linear(fc_1, num_classes))

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
    return model


def get_SixLayerOctConvNet(alpha, freq_ratio, hidden_channels, C, H, W, fc_1, D_out):
    """
    Returns stacked 4 layer octConvNet as in modules, using stacked format.

    """
    terminal_alpha = [0] * len(alpha)
    model = nn.Sequential(
        nn.Conv2d(C, hidden_channels, (3, 3), padding=1), # First layer is conv2D as in paper
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), terminal_alpha, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (2, 2), alpha, alpha, freq_ratio=freq_ratio, padding=0, stride=2), # Downsamples
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, alpha, freq_ratio=freq_ratio, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (2, 2), alpha, terminal_alpha, freq_ratio=freq_ratio, padding=0, stride=2), # Downsamples
        nn.ReLU(),
        Flatten(),
        nn.Linear(hidden_channels * (H // 4) * (W // 4), fc_1),
        nn.ReLU(),
        nn.Linear(fc_1, D_out)
        )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, OctConv2d):
            for _, layer in m.conv_dict.items():
                if layer is not None:
                    nn.init.kaiming_normal_(layer.weight)
            
    return model