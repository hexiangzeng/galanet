# -*- coding: utf-8 -*-
"""
# @Time    : May/23/2020
# @Author  : zhx
"""

import torch
from torch import nn
import torch.nn.functional as F
from network_architecture.Unet import StackConvNormNonlin
from network_architecture.initialize_network import InitWeights_He


class DilatedConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, conv_dilation=1, norm_op=nn.InstanceNorm3d, nonlin_op=nn.ReLU):
        super(DilatedConvNormNonlin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=conv_dilation, dilation=conv_dilation, bias=True)
        self.norm = norm_op(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.nonlin = nonlin_op(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.nonlin(self.norm(x))


class StackDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_lists=(1, 2, 3), basic_block=DilatedConvNormNonlin,
                 short_cut=False):
        super(StackDilatedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilated_conv_list =nn.Sequential(
            *([basic_block(in_channels, out_channels, conv_dilation=dilation_lists[0])] +
              [basic_block(out_channels, out_channels, dilation) for dilation in dilation_lists[1:]])
        )
        self.short_cut = short_cut
        if short_cut and in_channels!=out_channels:
            self.short_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
            self.short_norm = nn.InstanceNorm3d(out_channels, affine=True)
            self.short_nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dilated_conv_list(x)
        if self.short_cut and self.in_channels != self.out_channels:
            return self.short_nonlin(out+self.short_norm(self.short_conv(x)))
        else:
            return out


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='trilinear', align_corners=True):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class GALANet(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size=None, features_num_lists=(16, 32, 64, 128),
                 decoder_num_convs=2, trilinear_upsample=True):
        super(GALANet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.features_num_lists = features_num_lists
        self.encoder = []
        self.down = []
        self.up = []
        self.decoder = []
        in_features = 1
        out_features = features_num_lists[0]
        for d in range(len(features_num_lists)-1):
            self.encoder.append(StackDilatedBlock(in_features, out_features))
            self.down.append(nn.MaxPool3d(2, 2))
            in_features = features_num_lists[d]
            out_features = features_num_lists[d+1]

        # bottleneck
        self.encoder.append(StackDilatedBlock(in_features, out_features))

        final_features = features_num_lists[0]
        for u in range(len(features_num_lists)-1):
            down_features = self.encoder[-(u+1)].out_channels
            skip_features = self.encoder[-(u+2)].out_channels
            concat_features = 2*skip_features
            final_out_features = skip_features
            if trilinear_upsample:
                self.up.append(nn.Sequential(
                    Upsample(scale_factor=2, mode='trilinear'),
                    nn.Conv3d(down_features, skip_features, 1, 1, 0, 1)
                ))
            else:
                self.up.append(nn.ConvTranspose3d(down_features, down_features, 2, 2, bias=False))
            self.decoder.append(StackConvNormNonlin(concat_features, final_out_features, num_convs=decoder_num_convs,
                                                    basic_block=DilatedConvNormNonlin, short_cut=True))

        self.class_conv = nn.Conv3d(final_features, num_classes, 1)

        self.encoder = nn.ModuleList(self.encoder)
        self.down = nn.ModuleList(self.down)
        self.decoder = nn.ModuleList(self.decoder)
        self.up = nn.ModuleList(self.up)
        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        # print("data shape: {}".format(x.shape))
        skips = []
        for d in range(len(self.encoder)-1):
            x = self.encoder[d](x)
            # print("stage {}: dilated conv output x shape:{}".format(d, x.shape))
            skips.append(x)
            x = self.down[d](x)
        x = self.encoder[-1](x)

        for u in range(len(self.decoder)):
            x = self.up[u](x)
            if skips[-(u+1)].shape != x.shape:
                # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
                x = F.pad(x, [abs(skips[-(u+1)].shape[-1]-x.shape[-1]), 0,
                              abs(skips[-(u+1)].shape[-2]-x.shape[-2]), 0,
                              abs(skips[-(u+1)].shape[-3]-x.shape[-3]), 0])
            # print("{}, {}".format(skips[-(u+1)].shape, x.shape))
            x = torch.cat((torch.sigmoid(x)*skips[-(u+1)], x), dim=1)
            x = self.decoder[u](x)
        x = self.class_conv(x)
        return x


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    x = torch.randn(1, 1, 128, 192, 192)
    net = GALANet(1, 6, x.shape[2:], [16, 32, 64, 128]).cuda()
    print(x.shape)
    x = x.cuda()
    print(net)
    out = net.forward(x)
    print(out.shape)

    # [128, 196, 196], [16, 64, 128, 224]
