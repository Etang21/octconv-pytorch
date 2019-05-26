"""
Implementation of the Octave Convolution module in PyTorch.

OctConv2d is our building block. Takes in x_h, x_l and returns out_h and out_l.

OctConv2dStackable implements some tensor rearrangements to achieve the same functionality
with just one input and one output.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OctConv2d(nn.Module):
    """
    OctConv2D layer which takes in x_h and x_l and returns out_h and out_l.
    
    Building block layer for OctConv2DStackable.
    """
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
    
        
    def forward(self, x_h, x_l):
        """
        Computes a forward pass for the OctConv layer.
        
        Inputs:
        - x_h: Input high-frequency data, of shape (N, (1 - alpha_in) * C, H, W)
        - x_l: Input low-frequency data, of shape(N, alpha_in * C, H / 2, W / 2)
            - If alpha_in = 0, x_l can be anything
        
        Returns a tuple of:
        - out_h: High-frequency outputs, of shape (N, (1 - alpha_out) * F, H', W')
        - out_l: Low-frequency outputs, of shape (N, alpha_out * F, H' / 2, W' / 2) 
                    or None if alpha_out = 0
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

    def decompose_input(self, input):
        """
        Disassembles Numpy array into x_l and x_h for consumption by OctConv2d.
        
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
        """
        x_h, x_l_aggregate = torch.split(input, [self.in_channels_h, self.in_channels_l//4], dim=1)
        x_l_01, x_l_23  = torch.split(x_l_aggregate, x_l_aggregate.shape[2]//2, dim=2)
        x_l_0, x_l_1 = torch.split(x_l_01, x_l_01.shape[3]//2, dim=3)
        x_l_2, x_l_3 = torch.split(x_l_23, x_l_23.shape[3]//2, dim=3)
        x_l = torch.cat((x_l_0, x_l_1, x_l_2, x_l_3), dim=1)
        return x_h, x_l
    
    def compose_output(self, out_h, out_l):
        """
        Reassembles x_h and x_l into an output numpy array (the h-2x2_l shape tensor).
        
        Uses the reverse of the method documented in decompose_input
        """
        if out_l is not None:
            out_l_0, out_l_1, out_l_2, out_l_3 = torch.split(out_l, [out_l.shape[1]//4, out_l.shape[1]//4, out_l.shape[1]//4, out_l.shape[1]//4], dim=1)
            out_l_01 = torch.cat((out_l_0, out_l_1), dim=3)
            out_l_23 = torch.cat((out_l_2, out_l_3), dim=3)
            out_l_aggregate = torch.cat((out_l_01, out_l_23), dim=2)
            out = torch.cat((out_h, out_l_aggregate), dim=1)
            return out
        else:
            return out_h

class OctConv2dStackable(OctConv2d):
    """
    Wrapper for OctConv2d. Takes in one numpy array as input and outputs one as output.
    
    Useful for adding to sequential modules!
    """
    
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
        """
        super().__init__(in_channels, out_channels, kernel_size, 
                         alpha_in, alpha_out, stride=stride, padding=padding)
        
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

        Returns:
        - Analogously structured numpy array containing out_h and out_l.
        """
        # Decompose input into x_l and x_h using the food-in-fridge method
        x_h, x_l = super().decompose_input(input)

        # Apply Octave Convolutions
        out_h, out_l = super().forward(x_h, x_l)
        
        # Rebuild the h-2x2_l shape tensor
        return super().compose_output(out_h, out_l)
    
class OctConv2dBN(OctConv2d):
    """
    Implements an OctConv2d layer with batchnormalization for each feature map.
    Also uses the tricks from OctConv2dStackable to take one input and one output.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride=1, padding=0):
        """
        Initializes OctConv2d layer with batchnorm for both frequencies.
        Further documentation in OctConv2d class.
        """
        super().__init__(in_channels, out_channels, kernel_size, 
                 alpha_in, alpha_out, stride=stride, padding=padding)
        self.bn_h = nn.BatchNorm2d(int(in_channels * (1 - alpha_in)))
        self.bn_l = nn.BatchNorm2d(int(in_channels * alpha_in))

    
    def forward(self, input):
        """
        Computes a forward pass for the OctConv layer.
        Applies decomposition -> batchnorm -> octconv -> composition.
        """
        x_h, x_l = self.decompose_input(input)
        
        if self.bn_h.num_features > 0:
            x_h = self.bn_h(x_h)
        if self.bn_l.num_features > 0:
            x_l = self.bn_l(x_l)
            
        out_h, out_l = super().forward(x_h, x_l)
        return self.compose_output(out_h, out_l)
    
        
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
        nn.ReLU(),
        nn.MaxPool2d(2),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, 0, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(hidden_channels * (H // 4) * (W // 4), D_out)
        )
    return model

def get_stacked_4BN(alpha, hidden_channels, C, H, W, D_out):
    """
    Returns stacked 4 layer octConvNet with Batchnorm at each OctConv layer.

    """
    model = nn.Sequential(
        OctConv2dBN(C, hidden_channels, (3, 3), 0, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, alpha, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dBN(hidden_channels, hidden_channels, (3, 3), alpha, 0, stride=1, padding=1),
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


def get_SixLayerOctConvNet(alpha, hidden_channels, C, H, W, fc_1, D_out):
    """
    Returns stacked 4 layer octConvNet as in modules, using stacked format.

    """
    model = nn.Sequential(
        nn.Conv2d(C, hidden_channels, (3, 3), padding=1), # First layer is conv2D as in paper
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), 0, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (2, 2), alpha, alpha, padding=0, stride=2), # Downsamples
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (3, 3), alpha, alpha, stride=1, padding=1),
        nn.ReLU(),
        OctConv2dStackable(hidden_channels, hidden_channels, (2, 2), alpha, 0, padding=0, stride=2), # Downsamples
        nn.ReLU(),
        Flatten(),
        nn.Linear(hidden_channels * (H // 4) * (W // 4), fc_1),
        nn.ReLU(),
        nn.Linear(fc_1, D_out)
        )
    
    # TODO: Add Kaiming-He initialization code here?
    
    return model