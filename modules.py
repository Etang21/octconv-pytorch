"""
Implementation of the Octave Convolution module in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OctConv2d(nn.Module):
    
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
        in_channels_l, out_channels_l = int(in_channels_l), int(out_channels_l)
        in_channels_h, out_channels_h = in_channels - in_channels_l, out_channels - out_channels_l
        
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
    
    def forward(self, x_h, x_l):
        """
        Computes a forward pass for the OctConv layer.
        
        Inputs:
        - x_h: Input high-frequency data, of shape (N, (1 - alpha_in) * C, H, W)
        - x_l: Input low-frequency data, of shape(N, alpha_in * C, H / 2, W / 2)
            - If alpha_in = 0, x_l can be anything
        
        Returns a tuple of:
        - out_h: High-frequency outputs, of shape (N, (1 - alpha_out) * F, H', W')
        - out_l: Low-frequency outputs, of shape (N, alpha_out * F, H' / 2, W' / 2) or None if alpha_out = 0
        """
        out_h, out_l = None, None
        
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
        return out_h, out_l
    
def flatten(x):
    """
    Flsttens input vector, preserves first dimension
    
    Code taken from Assignment 2, Pytorch notebook
    """
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



class FourLayerOctConvNet(nn.Module):
    """
    Four layer octconv net for testing.
    
    Architecture: [Octconv -> ReLU -> OctConv -> ReLU -> Pool]*2 -> FC
    - OctConv layers use 3 x 3 kernels with stride 1, padding 1
    - Pooling layers use 2 x 2 kernels with stride 2
    
    Implementation notes:
    # We have two outputs/inputs at each layer, for the low-frequency and high-frequency channels
    # We can't use nn.sequential because sequential only takes one input at each stage
    # So we subclass nn.module
    """
    
    def __init__(self, alpha, hidden_channels, C, H, W, D_out):
        """
        Initialize a four-layer Octconv network.
        
        Parameters:
        alpha (float): proportion of channels that are low-frequency in hidden layers
        C (int): number of input channels
        H (int): height of input data
        W (int): width of input data
        F (int): number of filters in each hidden layer
        D_out (int): length of output vector
        """
        super().__init__()
        self.alpha = alpha
        self.oc1 = OctConv2d(C, hidden_channels, (3, 3), 0, self.alpha, stride=1, padding=1)
        self.oc2 = OctConv2d(hidden_channels, hidden_channels, (3, 3), self.alpha, self.alpha, stride=1, padding=1)
        self.oc3 = OctConv2d(hidden_channels, hidden_channels, (3, 3), self.alpha, self.alpha, stride=1, padding=1)
        self.oc4 = OctConv2d(hidden_channels, hidden_channels, (3, 3), self.alpha, 0, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(hidden_channels * (H // 4) * (W // 4), D_out)
        # Question: Do we need to initialize the weights of oc layers under the OctConv module?
    
    def forward(self, x):
        x_h, x_l = self.oc1(x, None) # alpha_in = 0
        x_h, x_l = F.relu(x_h), F.relu(x_l)
        x_h, x_l = self.oc2(x_h, x_l)
        x_h, x_l = F.relu(x_h), F.relu(x_l)
        x_h, x_l = F.max_pool2d(x_h, (2, 2), stride=2), F.max_pool2d(x_l, (2, 2), stride=2)
        
        x_h, x_l = self.oc3(x_h, x_l)
        x_h, x_l = F.relu(x_h), F.relu(x_l)
        x_h, _ = self.oc4(x_h, x_l) # alpha_out = 0
        x_h = F.relu(x_h)
        x_h = F.max_pool2d(x_h, (2, 2), stride=2)
        x_h = flatten(x_h)
        
        out = self.fc1(x_h)
        return out