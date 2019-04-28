"""
File to store the modules we write
"""

import torch
import torch.nn as nn

class OctConv2d(nn.Module):
    # Currently just a test of basic conv behavior
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        # TODO: Add stride, padding, and other params
    
    def forward(self, x):
        y = self.conv1(x)
        return y