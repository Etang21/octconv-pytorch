"""
Implementation of the Octave Convolution module in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OctConv2dStackable(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride=1, padding=0):
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
            
        # TODO: Add other params that nn.Conv2d also supports
        """
        super().__init__()
        
        # Compute number of high-freq and low-freq channels
        in_channels_l, out_channels_l = alpha_in * in_channels, alpha_out * out_channels
        assert float(in_channels_l).is_integer(), "Incompatible in_channels, alpha_in"
        assert float(out_channels_l).is_integer(), "Incompatible out_channels, alpha_out"
        # Additional constraint for now for stacking. TODO: Fix in general
        assert in_channels_l % 4 == 0, "Please use a multiple of four low freq input channels"
        assert out_channels_l % 4 == 0, "Please use a multiple of four low freq input channels"
        in_channels_l, out_channels_l = int(in_channels_l), int(out_channels_l)
        in_channels_h, out_channels_h = in_channels - in_channels_l, out_channels - out_channels_l

        # Remember these for forward purposes - John
        self.in_channels_l, self.in_channels_h, self.out_channels_l, self.out_channels_h = in_channels_l, in_channels_h, out_channels_l, out_channels_h
        
        # Check for low-freq outputs or inputs
        self.has_in_l = in_channels_l > 0
        self.has_out_l = out_channels_l > 0
        
        # Create conv layers
        # self.conv_hh stores weight W^{H -> H}. Analogous notation for other conv layers.
        self.conv_hh = nn.Conv2d(in_channels_h, out_channels_h, kernel_size, stride=stride, padding=padding)
        if self.has_in_l and self.has_out_l:
            self.conv_ll = nn.Conv2d(in_channels_l, out_channels_l, kernel_size, stride=stride, padding=padding)
        if self.has_in_l:
            self.conv_lh = nn.Conv2d(in_channels_l, out_channels_h, kernel_size, stride=stride, padding=padding)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if self.has_out_l:
            self.conv_hl = nn.Conv2d(in_channels_h, out_channels_l, kernel_size, stride=stride, padding=padding)
            self.pool = nn.AvgPool2d((2, 2), stride=2)
    
    def forward(self, input):
        """
        Computes a forward pass for the OctConv layer.
        
        Inputs:
        (MODIFIED -- Now takes input in the form x_h, (2x2 square of x_l's))
        (x_l is cut into four pieces N wise, then stacked as follows, where we're observing from
        (the bottom zero index of each tensor:
        ( 0 1 -- W
        ( 2 3
        (  |
        (  H

        - x_h: Input high-frequency data, of shape (N, (1 - alpha_in) * C, H, W)
        - x_l: Input low-frequency data, of shape(N, alpha_in * C, H / 2, W / 2)
            - If alpha_in = 0, x_l can be anything
        
        Returns a tuple of:
        - out_h: High-frequency outputs, of shape (N, (1 - alpha_out) * F, H', W')
        - out_l: Low-frequency outputs, of shape (N, alpha_out * F, H' / 2, W' / 2) or None if alpha_out = 0
        """
        out_h, out_l = None, None
        
        # Build up
        x_h, x_l_aggregate = torch.split(input, [self.in_channels_h, self.in_channels_l//4], dim=1)
        x_l_01, x_l_23  = torch.split(x_l_aggregate, x_l_aggregate.shape[2]//2, dim=2)
        x_l_0, x_l_1 = torch.split(x_l_01, x_l_01.shape[3]//2, dim=3)
        x_l_2, x_l_3 = torch.split(x_l_23, x_l_23.shape[3]//2, dim=3)
        x_l = torch.cat((x_l_0, x_l_1, x_l_2, x_l_3), dim=1)

        # Compute out_h
        out_h = self.conv_hh(x_h)
        assert out_h.shape[2] % 2 == 0, "OctConv output width not divisible by 2"
        assert out_h.shape[3] % 2 == 0, "OctConv output height not divisible by 2"
        if self.has_in_l:
            out_h += self.upsample(self.conv_lh(x_l))
            
        # Compute out_l
        if self.has_out_l:
            out_l = self.conv_hl(self.pool(x_h))
            if self.has_in_l:
                out_l += self.conv_ll(x_l)
        
        # Rebuild the h-2x2_l shape tensor
        if out_l is not None:
            out_l_0, out_l_1, out_l_2, out_l_3 = torch.split(out_l, [out_l.shape[1]//4, out_l.shape[1]//4, out_l.shape[1]//4, out_l.shape[1]//4], dim=1)
            out_l_01 = torch.cat((out_l_0, out_l_1), dim=3)
            out_l_23 = torch.cat((out_l_2, out_l_3), dim=3)
            out_l_aggregate = torch.cat((out_l_01, out_l_23), dim=2)
            out = torch.cat((out_h, out_l_aggregate), dim=1)
            return out
        else:
            return out_h
    
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

def get_stacked_4(alpha, hidden_channels, C, H, W, D_out):
    """
    Returns stacked 4 layer octConvNet as in modules, using stacked format.

    """
    model = nn.Sequential(
        OctConv2dStackable(C, hidden_channels, (3, 3), 0, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, alpha, stride=1, padding=1),
        nn.ReLu(),
        nn.MaxPool2d(2),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, 0, stride=1, padding=1),
        nn.ReLu(),
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
        nn.Conv2d(channels[1], channels[2], (3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(channels[2], channels[3], (2, 2), padding=0, stride=2), # Downsamples
        nn.ReLU(),
        nn.Conv2d(channels[3], channels[4], (3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(channels[4], channels[5], (3, 3), padding=1),
        nn.ReLU(),
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