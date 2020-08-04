# -*- coding: utf-8 -*-
"""
# @Time    : May/30/2020
# @Author  : zhx
"""

import torch
import torch
from torch import nn
import torch.nn.functional as F
from network_architecture.initialize_network import InitWeights_He
from network_architecture.GALANet import Upsample


class ConvBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBasic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.norm = nn.BatchNorm3d(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.nonlin(self.conv(x))  # 原论文是 conv3x3x3, relu, batch_normalization
        return self.norm(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, basic_conv=ConvBasic):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1x1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm3d(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.resblock = nn.Sequential(
            *[basic_conv(out_channels, out_channels) for _ in range(num_convs)]
        )

    def forward(self, x):
        output1 = self.norm(self.relu(self.conv1x1x1(x)))
        output2 = self.resblock(output1)
        return output2+output1


class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvDown, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm3d(out_channels, eps=1e-5, affine=True, momentum=0.1)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return self.norm(output)


class APLSNet(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, features_num_lists=(32, 40, 48, 56)):
        super(APLSNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features_num_lists = features_num_lists
        self.patch_size = patch_size

        in_features = 1
        out_features = features_num_lists[0]
        self.convdown = ConvDown(in_features, out_features)
        self.encoder = []
        self.decoder = []
        for new_features in features_num_lists[1:]:
            in_features = out_features
            out_features = new_features
            self.encoder.append(ResidualBlock(in_features, out_features, num_convs=2))
        for u in range(len(features_num_lists)-1):
            in_features = features_num_lists[-(u+1)] + features_num_lists[-(u+2)]
            out_features = features_num_lists[-(u+2)]
            self.decoder.append(ResidualBlock(in_features, out_features, num_convs=2))

        self.up = Upsample(scale_factor=2, mode='trilinear')
        in_features = 1+features_num_lists[0]
        out_features = num_classes
        self.classblock = ResidualBlock(in_features, out_features)
        self.conv1x1x1 = nn.Conv3d(out_features, out_features, 1, 1, 0, 1, bias=True)

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

        self.apply(InitWeights_He(1e-2))

    def forward(self, inputs):
        convdown = self.convdown(inputs)
        skip = []
        x = convdown
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            skip.append(x)

        for u in range(len(self.decoder)-1):
            x = torch.cat([x, skip[-(u+2)]], dim=1)
            x = self.decoder[u](x)

        x = torch.cat([convdown, x], dim=1)
        x = self.decoder[-1](x)  # channels 32

        x = self.up(x)
        if inputs.shape[2:] != x.shape[2:]:
            diff = [i-j for i, j in zip(x.shape[2:], inputs.shape[2:])]
            D, H, W = x.shape[2:]
            x = x[:, :, :D-diff[0], :H-diff[1], :W-diff[2]]
            # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
            x = F.pad(x, [abs(inputs.shape[-1] - x.shape[-1]), 0,
                          abs(inputs.shape[-2] - x.shape[-2]), 0,
                          abs(inputs.shape[-3] - x.shape[-3]), 0])
        # print(inputs.shape, x.shape)
        x = torch.cat([inputs, x], dim=1)
        x = self.classblock(x)

        return self.conv1x1x1(x)


if __name__ == '__main__':
    net = APLSNet(1, 6, [32, 40, 48, 56]).cuda()
    x = torch.randn(1, 1, 54, 74, 127)
    print(x)
    x = x.cuda()
    print(net)
    out = net.forward(x)
    print(out.shape)
