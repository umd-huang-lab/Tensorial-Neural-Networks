import torch
import torch.nn as nn
import math
import numpy as np
from layers import *

def conv3x3(in_planes, out_planes, decompose_type=None, cr=None, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    if decompose_type == "RCP":
        return RCP_O4_Conv2d(in_planes, out_planes, 3, 8, cr=cr, stride=stride,
                             padding=dilation, groups=groups, dilation=dilation)
    elif decompose_type == "RTT":
        return RTT_O4_Conv2d(in_planes, out_planes, 3, 8, cr=cr, stride=stride,
                             padding=dilation, groups=groups, dilation=dilation)
    elif decompose_type == "RTR":
        return RTR_O4_Conv2d(in_planes, out_planes, 3, 8, cr=cr, stride=stride,
                             padding=dilation, groups=groups, dilation=dilation)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, decompose_type, cr=None, bias=False):
    """1x1 convolution"""
    if decompose_type == "RCP":
        return RCP_O4_Linear(in_planes, out_planes, cr=cr, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, decompose_type=None, cr=None, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, decompose_type=decompose_type, cr=cr, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, decompose_type=decompose_type, cr=cr)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, decompose_type=None, cr=None, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, decompose_type=decompose_type, cr=cr, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, decompose_type=decompose_type, cr=cr, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, decompose_type=decompose_type, cr=cr, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, decompose_type, cr, num_classes=1000, channel=3):
        super(ResNet, self).__init__()
        self.inplanes = 64

        #self.conv1 = conv7x7(channel, self.inplanes, decompose_type=decompose_type, cr=cr, stride=2)
        self.conv1 = nn.Conv2d(channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, decompose_type, cr, 64, layers[0])
        self.layer2 = self._make_layer(block, decompose_type, cr, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, decompose_type, cr, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, decompose_type, cr, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, decompose_type, cr, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, decompose_type, cr, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, decompose_type, cr=cr))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

def ResNet_(name="ResNet50",type="RCP", cr=None):
    if name == "ResNet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], type, cr)
    elif name == "ResNet34":
        return ResNet(BasicBlock, [3, 4, 6, 3], type, cr)
    elif name == "ResNet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], type, cr)
    elif name == "ResNet101":
        return ResNet(Bottleneck, [3, 4, 23, 3], type, cr)
    elif name == "ResNet152":
        return ResNet(Bottleneck, [3, 8, 36, 3], type, cr)
