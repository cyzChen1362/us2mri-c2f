"""slice_matching / encoder

Signatures only. Removed in the public release (desensitised); see the module docstring.
"""
import torch
from torch import nn

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        raise NotImplementedError('Removed in the public release (desensitised); see the module docstring.')

    def forward(self, x):
        raise NotImplementedError('Removed in the public release (desensitised); see the module docstring.')

class ResNet50Encoder(nn.Module):

    def __init__(self, groups=32, *, normalization='group', weights=None):
        raise NotImplementedError('Removed in the public release (desensitised); see the module docstring.')

    def forward(self, x):
        raise NotImplementedError('Removed in the public release (desensitised); see the module docstring.')
