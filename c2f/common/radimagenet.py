"""RadImageNet-compatible ResNet50; stage one fixes BatchNorm running statistics."""
import torch
from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

class ResNet50Encoder(nn.Module):

    def __init__(self, groups=32, *, normalization='batch', weights=None):
        super().__init__()
        if normalization == 'group':
            norm_layer = lambda channels: nn.GroupNorm(groups, channels)
        elif normalization == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError('Unknown normalization: ' + normalization)
        layers = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), norm_layer(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1)]
        inplanes = 64
        for planes, count, stride in zip((64, 128, 256, 512), (3, 4, 6, 3), (1, 2, 2, 2)):
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, 1, stride=stride, bias=False), norm_layer(planes * 4))
            blocks = [Bottleneck(inplanes, planes, stride, downsample, norm_layer)]
            inplanes = planes * 4
            blocks.extend((Bottleneck(inplanes, planes, norm_layer=norm_layer) for _ in range(count - 1)))
            layers.append(nn.Sequential(*blocks))
        self.backbone = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if weights is not None:
            if normalization != 'batch':
                raise ValueError('RadImageNet checkpoint requires the original BatchNorm architecture.')
            state = torch.load(weights, map_location='cpu', weights_only=True)
            self.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, x):
        for module in list(self.backbone.children())[:-1]:
            x = module(x)
        return x
