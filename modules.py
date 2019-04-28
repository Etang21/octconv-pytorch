"""
Implementation of the Octave Convolution module in PyTorch.
"""

import torch
import torch.nn as nn

class OctConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in, alpha_out):
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
            
        # TODO: Add stride, padding, and other params
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
        self.conv_hh = nn.Conv2d(in_channels_h, out_channels_h, kernel_size)
        if self.has_in_l and self.has_out_l:
            self.conv_ll = nn.Conv2d(in_channels_l, out_channels_l, kernel_size)
        if self.has_in_l:
            pass
            # TODO: Initialize self.conv_lh
        if self.has_out_l:
            pass
            # TODO: Initialize self.conv_hl
    
    def forward(self, x_h, x_l):
        """
        Computes a forward pass for the OctConv layer.
        
        Inputs:
        - x_h: Input high-frequency data, of shape (N, (1 - alpha_in) * C, H, W)
        - x_l: Input low-frequency data, of shape(N, alpha_in * C, H / 2, W / 2)
        
        Returns a tuple of:
        - out_h: High-frequency outputs, of shape (N, (1 - alpha_out) * F, H, W)
        - out_l: Low-frequency outputs, of shape (N, alpha_out * F, H / 2, W / 2) or None if alpha_out = 0
        """
        out_h, out_l = None, None
        out_h = self.conv_hh(x_h)
        if self.has_in_l and self.has_out_l:
            out_l = self.conv_ll(x_l)
        if self.has_in_l:
            pass
            # TODO: Implement forward pass from l to h, add to out_h
        if self.has_out_l:
            pass
            # TODO: Implement forward pass from h to l, add to out_l
        return out_h, out_l