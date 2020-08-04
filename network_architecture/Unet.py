# -*- coding: utf-8 -*-
"""
# @Time    : May/23/2020
# @Author  : zhx
"""

import torch
from torch import nn
import torch.nn.functional as F
from network_architecture.initialize_network import InitWeights_He


class ConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNormNonlin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.norm = nn.BatchNorm3d(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.nonlin(self.conv(x)) # 原论文是 conv3x3x3, relu, batch_normalization
        return self.norm(x)


class StackConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, basic_block=ConvNormNonlin, short_cut=False):
        super(StackConvNormNonlin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.Sequential(
            *([basic_block(in_channels, out_channels)] +
              [basic_block(out_channels, out_channels) for _ in range(num_convs-1)])
        )
        self.short_cut = short_cut
        if short_cut and in_channels!=out_channels:
            self.short_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
            self.short_norm = nn.InstanceNorm3d(out_channels, affine=True)
            self.short_nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.blocks(x)
        if self.short_cut and self.in_channels != self.out_channels:
            return self.short_nonlin(out+self.short_norm(self.short_conv(x)))
        else:
            return out


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, features_num_lists=(16, 32, 64, 128, 256),
                 num_convs_per_stage=2):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.features_num_lists = features_num_lists
        # input size is (128, 160, 160)
        self.conv_blocks_analysis = []
        self.conv_blocks_synthesis = []
        self.down = []
        self.up = []
        input_features = in_channels
        output_features = features_num_lists[0]
        for idx in range(len(features_num_lists)-1):
            self.conv_blocks_analysis.append(StackConvNormNonlin(input_features, output_features, num_convs_per_stage))
            if idx < len(features_num_lists)-1:
                self.down.append(nn.MaxPool3d(2))
            input_features = features_num_lists[idx]
            output_features = features_num_lists[idx+1]

        # bottleneck conv
        self.conv_blocks_analysis.append(StackConvNormNonlin(input_features, output_features, num_convs_per_stage))

        final_out_features = self.conv_blocks_analysis[-2].out_channels
        for u in range(len(features_num_lists)-1):
            down_features = self.conv_blocks_analysis[-(u+1)].out_channels
            skip_features = self.conv_blocks_analysis[-(u+2)].out_channels
            concat_features = down_features+skip_features
            final_out_features = skip_features
            self.up.append(nn.ConvTranspose3d(down_features, down_features, 2, 2, bias=False))
            self.conv_blocks_synthesis.append(nn.Sequential(
                StackConvNormNonlin(concat_features, final_out_features, 1),
                StackConvNormNonlin(final_out_features, final_out_features, num_convs_per_stage-1)
            ))

        self.class_conv = nn.Conv3d(final_out_features, num_classes, kernel_size=1)

        self.conv_blocks_analysis = nn.ModuleList(self.conv_blocks_analysis)
        self.conv_blocks_synthesis = nn.ModuleList(self.conv_blocks_synthesis)
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        # print(x.shape)
        skips = []
        for d in range(len(self.conv_blocks_analysis)-1):
            x = self.conv_blocks_analysis[d](x)
            skips.append(x)
            x = self.down[d](x)
        x = self.conv_blocks_analysis[-1](x)

        for u in range(len(self.conv_blocks_synthesis)):
            x = self.up[u](x)
            if skips[-(u+1)].shape != x.shape:
                # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
                x = F.pad(x, [abs(skips[-(u+1)].shape[-1]-x.shape[-1]), 0,
                              abs(skips[-(u+1)].shape[-2]-x.shape[-2]), 0,
                              abs(skips[-(u+1)].shape[-3]-x.shape[-3]), 0])
            x = torch.cat((skips[-(u+1)], x), dim=1)
            x = self.conv_blocks_synthesis[u](x)
        x = self.class_conv(x)
        return x


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    net = Unet(1, 6, 2, [16, 32, 64, 128, 128]).cuda()
    x = torch.randn(1, 1, 64, 92, 92)
    print(x)
    x = x.cuda()
    print(net)
    # writer = SummaryWriter('/media/zhx/My Passport/lung_lobe_seg/galaNet_trained_models/3DUnet/logger')
    # with writer:
    #     writer.add_graph(net, x)
    out = net.forward(x)
    print(out)
