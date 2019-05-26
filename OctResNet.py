"""
Implementation of ResNet with Octave Convolution.
Copied and modified from Pytorch source ResNet implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#import modules.OctConv2d.OctConv2dStackable as Oct2d
#import modules.OctConv2dBN as Oct2dBN
import modules

Oct2d = modules.OctConv2dStackable
OctConv2d = modules.OctConv2d
Oct2dBN = modules.OctConv2dBN


def conv3x3(in_planes, out_planes, alpha_in=.25, alpha_out=.25, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, 3, alpha_in, alpha_out, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, alpha_in=.25, alpha_out=.25, stride=1):
    """1x1 convolution"""
    return Oct2d(in_planes, out_planes, 1, alpha_in, alpha_out, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, alpha_in=.25, alpha_mid=.25, alpha_out=.25, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.dilation = dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1bn = Oct2dBN(inplanes, planes, (3, 3), alpha_in, alpha_out, stride=stride, padding=dilation)

        self.relu = nn.ReLU(inplace=True)

        self.conv2bn = Oct2dBN(planes, planes, (3, 3), alpha_out, alpha_out, padding=dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        print(x.shape)
        out = self.conv1bn(x)
        out = self.relu(out)
        print(out.shape)
        out = self.conv2bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha_in=.25, alpha_mid=.25, alpha_out=.25, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1bn = Oct2dBN(in_planes, width, (1, 1), alpha_in, alpha_mid, stride=stride, padding=dilation)
        self.conv2bn = Oct2dBN(width, width, (3, 3), alpha_mid, alpha_mid, stride=stride, padding=dilation)
        #conv3x3(width, width, stride, groups, dilation)
        
        #self.conv3bn = conv1x1(width, planes * self.expansion)
        self.conv3bn = Oct2dBN(width, planes * self.expansion, (1, 1), alpha_mid, alpha_out, stride=stride, padding=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1bn1(x)
        out = self.relu(out)

        out = self.conv2bn(out)
        out = self.relu(out)

        out = self.conv3bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class OctaveResNet(nn.Module):


    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, alpha=.25,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(OctaveResNet, self).__init__()
        self.alpha = alpha
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #Convolve, break, then normalize, pool, etc.
        self.conv1 = Oct2d(3, self.inplanes, (7, 7), 0, self.alpha, stride=2, padding=3)
        #self.conv1bn = Oct2dBN(inplanes, planes, (3, 3), alpha_in, alpha_out, stride=stride, padding=dilation)
        self.bnLow = norm_layer(int(self.inplanes*self.alpha))
        self.bnHigh = norm_layer(int(self.inplanes*(1 - self.alpha)))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], final_layer=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, final_layer=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #conv1x1(self.inplanes, planes * block.expansion, stride),
                #norm_layer(planes * block.expansion),
                Oct2dBN(self.inplanes, planes * block.expansion, (1, 1), self.alpha, self.alpha, stride)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.alpha, self.alpha, self.alpha, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            alpha = self.alpha
            if final_layer:
                alpha=0
            layers.append(block(self.inplanes, planes, alpha_out=alpha, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
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
        in_channels_h = int(input.shape[1] * (1 - self.alpha)/(1 - 3*self.alpha/4))
        in_channels_l = int(input.shape[1] * (self.alpha)/(1 - 3*self.alpha/4))
        x_h, x_l_aggregate = torch.split(input, [in_channels_h, in_channels_l//4], dim=1)
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

    def forward(self, x):
        # Initial alpha split

        x = self.conv1(x)
        x_h, x_l = self.decompose_input(x)
        x_h = self.maxpool(self.relu(self.bnHigh(x_h)))
        x_l = self.maxpool(self.relu(self.bnLow(x_l)))

        x = self.compose_output(x_h, x_l)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def tinyoctresnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2,2,2,2], pretrained, progress, num_classes=200, alpha=.25, **kwargs)
    #num_classes=1000, zero_init_residual=False, alpha=.25, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):

def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = OctaveResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
**kwargs)