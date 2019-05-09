import torch
import torch.nn as nn
from modules import OctConv2d

def test_octconv_shapes():
    """A series of tests to ensure the shapes of our Octconv layers line up"""
    # Test output shapes for 1x1 convolution
    oc = OctConv2d(16, 16, (1, 1), 0.5, 0.5)
    input_h = torch.randn(128, 8, 32, 32)
    input_l = torch.randn(128, 8, 16, 16)
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 8, 32, 32), "Incorrect high-frequency output shape for OctConv2d"
    assert output_l.shape == (128, 8, 16, 16), "Incorrect low-frequency output shape for OctConv2d"
    
    
    # Test output shapes for 1x1 convolution
    oc = OctConv2d(16, 16, (1, 1), 0.5, 0.5)
    input_h = torch.randn(128, 8, 32, 32)
    input_l = torch.randn(128, 8, 16, 16)
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 8, 32, 32), "Incorrect high-frequency output shape for OctConv2d"
    assert output_l.shape == (128, 8, 16, 16), "Incorrect low-frequency output shape for OctConv2d"
    
    # Test output shapes with alpha_in != alpha_out
    oc = OctConv2d(16, 16, (1, 1), 0.5, 0.25)
    input_h = torch.randn(128, 8, 32, 32)
    input_l = torch.randn(128, 8, 16, 16)
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 12, 32, 32), "Incorrect high-frequency output shape for OctConv2d"
    assert output_l.shape == (128, 4, 16, 16), "Incorrect low-frequency output shape for OctConv2d"
    
    # Test output shapes with alpha_in != alpha_out and in_channels != out_channels
    oc = OctConv2d(16, 32, (1, 1), 0.5, 0.25)
    input_h = torch.randn(128, 8, 32, 32)
    input_l = torch.randn(128, 8, 16, 16)
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 24, 32, 32), "Incorrect high-frequency output shape for OctConv2d"
    assert output_l.shape == (128, 8, 16, 16), "Incorrect low-frequency output shape for OctConv2d"
    
    # Test output shapes with alpha_in = alpha_out = 0
    oc = OctConv2d(16, 32, (1, 1), 0, 0)
    input_h = torch.randn(128, 16, 32, 32)
    input_l = torch.randn(128, 0, 16, 16)
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 32, 32, 32), "Incorrect high-frequency output shape for OctConv2d"
    assert output_l is None, "Incorrect low-frequency output shape for OctConv2d"
    
    # Test output shapes with alpha_in = 0, alpha_out > 0 (imitates first layer)
    oc = OctConv2d(16, 32, (1, 1), 0, 0.25)
    input_h = torch.randn(128, 16, 32, 32)
    input_l = None
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 24, 32, 32), "Incorrect high-frequency output shape for OctConv2d"
    assert output_l.shape == (128, 8, 16, 16), "Incorrect low-frequency output shape for OctConv2d"
    
    # Test output shapes with padding and stride
    oc = OctConv2d(16, 32, (3, 3), 0.5, 0.5, stride=1, padding=1)
    input_h = torch.randn(128, 8, 32, 32)
    input_l = torch.randn(128, 8, 16, 16)
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 16, 32, 32), "Shape mismatch for stride=1, padding=1"
    assert output_l.shape == (128, 16, 16, 16), "Shape mismatch for stride=1, padding=1"
    
    # Test output shapes with stride to downsample
    oc = OctConv2d(16, 32, (2, 2), 0.5, 0.5, stride=2, padding=0)
    input_h = torch.randn(128, 8, 32, 32)
    input_l = torch.randn(128, 8, 16, 16)
    output_h, output_l = oc(input_h, input_l)
    assert output_h.shape == (128, 16, 16, 16), "Shape mismatch for stride=2, padding=0"
    assert output_l.shape == (128, 16, 8, 8), "Shape mismatch for stride=2, padding=0"

def test_octconv_as_conv():
    # Test that OctConv2d behaves like Conv2d when alpha_in = alpha_out = 0
    oc = OctConv2d(3, 32, (3, 3), 0, 0, stride=1, padding=1)
    conv = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
    input_h = torch.randn(128, 3, 32, 32)
    input_l = None

    conv.weight = oc.conv_hh.weight
    conv.bias = oc.conv_hh.bias

    output_h, output_l = oc(input_h, input_l)
    output_conv = conv(input_h)
    assert output_h.shape == output_conv.shape, "OctConv2d and Conv2d have different output shapes"
    assert torch.all(torch.eq(output_h, output_conv)), "OctConv2d and Conv2d have different outputs"
    