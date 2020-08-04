# -*- coding: utf-8 -*-
"""
# @Time    : Jun/24/2020
# @Author  : zhx
"""

import torch
from torch import nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate=0.2):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(num_input_features, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm3d(growth_rate)
        self.nonlin = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(x)
        if self.drop_rate > 0:
            out = F.dropout3d(out, self.drop_rate, training=self.training, inplace=True)
        out = self.nonlin(self.norm(out))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_input_features, num_layers=4, growth_rate=4, drop_rate=0.2):
        super(DenseBlock, self).__init__()
        self.denseblock = nn.Sequential()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features+i*growth_rate, growth_rate, drop_rate)
            self.denseblock.add_module('denselayer%d' % i, layer)

    def forward(self, x):
        x = self.denseblock(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.nonlin(self.norm(x))


class PDVNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, patch_size=(64, 512, 512), base_features=24,
                 layers_per_block=(5, 10, 10), growth_rate_per_block=(4, 8, 16)):
        super(PDVNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.layers_per_block = layers_per_block
        self.growth_rate_per_block = growth_rate_per_block

        self.convdownsample1 = nn.Conv3d(in_channels, base_features, kernel_size=5, stride=2, padding=2, bias=False)
        self.dfb1 = DenseBlock(base_features, layers_per_block[0], growth_rate_per_block[0])
        self.convblock1 = ConvBlock(base_features+layers_per_block[0]*growth_rate_per_block[0], base_features)
        self.final_out_conv1 = nn.Conv3d(base_features, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convdownsample2 = nn.Conv3d(base_features, base_features, kernel_size=3, stride=2, padding=1, bias=False)
        self.dfb2 = DenseBlock(base_features, layers_per_block[1], growth_rate_per_block[1])
        self.convblock2 = ConvBlock(base_features + layers_per_block[1] * growth_rate_per_block[1], base_features)
        self.final_out_conv2 = nn.Conv3d(base_features*2, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convdownsample3 = nn.Conv3d(base_features, base_features, kernel_size=3, stride=2, padding=1, bias=False)
        self.dfb3 = DenseBlock(base_features, layers_per_block[2], growth_rate_per_block[2])
        self.convblock3 = ConvBlock(base_features + layers_per_block[2] * growth_rate_per_block[2], base_features)
        self.final_out_conv3 = nn.Conv3d(base_features*3, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.convdownsample1(x)
        out1 = self.convblock1(self.dfb1(x))

        x = self.convdownsample2(out1)
        out2 = self.convblock2(self.dfb2(x))

        x = self.convdownsample3(out2)
        out3 = self.convblock3(self.dfb3(x))

        out2 = F.interpolate(out2, size=out1.shape[2:], mode='trilinear', align_corners=True)
        out3 = F.interpolate(out3, size=out1.shape[2:], mode='trilinear', align_corners=True)

        final_out1 = self.final_out_conv1(out1)
        final_out2 = self.final_out_conv2(torch.cat([out1, out2], dim=1))
        final_out3 = self.final_out_conv3(torch.cat([out1, out2, out3], dim=1))

        if self.training:
            return final_out1, final_out2, final_out3
        else:
            return final_out3


if __name__ == '__main__':
    x = torch.randn(1, 25, 5, 5, 5)
    print(x.shape)
    block = DenseBlock(x.shape[1], num_layers=5, growth_rate=4)
    out_ = block(x)
    print(out_.shape)

