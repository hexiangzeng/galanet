# -*- coding: utf-8 -*-
"""
# @Time    : Jun/23/2020
# @Author  : zhx
"""

import torch
from torch import nn
import torch.nn.functional as F
from network_architecture.Unet import StackConvNormNonlin
from network_architecture.initialize_network import InitWeights_He


class DilatedConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, conv_dilation=1,
                 norm_op=nn.InstanceNorm3d, nonlin_op=nn.ReLU):
        super(DilatedConvNormNonlin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1,
                              padding=conv_dilation, dilation=conv_dilation, bias=True)
        self.norm = norm_op(out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.nonlin = nonlin_op(inplace=True)
        self.short_cut = nn.Sequential()
        if in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),
                nn.InstanceNorm3d(out_channels, affine=True)
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out += self.short_cut(x)
        return self.nonlin(out)


class ResNeXtBottleneck(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_channels, out_channels, cardinality=1, stride=1):
        super(ResNeXtBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels//2, kernel_size=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels//2, affine=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels//2, out_channels, kernel_size=3,
                               stride=stride, padding=0, groups=cardinality, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )

    def forward(self, x):
        out = self.nonlin1(self.norm1(self.conv1(x)))
        out = self.nonlin2(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        short = self.shortcut(x)
        out += short
        return self.nonlin3(out)


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_lists=(1, 2, 3, 4), basic_block=DilatedConvNormNonlin):
        super(DilatedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilated_conv_list = nn.Sequential(
            *([basic_block(in_channels, out_channels, dilation_lists[0])] +
              [basic_block(out_channels, out_channels, dilation) for dilation in dilation_lists[1:]])
        )

    def forward(self, x):
        return self.dilated_conv_list(x)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='trilinear', align_corners=True):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class GALANetv2(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, features_num_lists=(64, 128, 256), trilinear_upsample=False,
                 decoder_ResNeXt_num=1):
        super(GALANetv2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.encoder = []
        self.down = []
        self.up = []
        self.decoder = []

        for d in range(len(features_num_lists)):
            in_features = features_num_lists[d-1] if d > 0 else 1
            out_features = features_num_lists[d]
            self.down.append(ResNeXtBottleneck(in_features, out_features, stride=2))
            self.encoder.append(DilatedBlock(out_features, out_features))

        final_out_features = None
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
                self.up.append(nn.ConvTranspose3d(down_features, skip_features, 2, 2, bias=False))

            self.decoder.append(nn.Sequential(
                *([ResNeXtBottleneck(concat_features, final_out_features)] +
                  [ResNeXtBottleneck(final_out_features, final_out_features) for _ in range(decoder_ResNeXt_num-1)])
            ))
        assert final_out_features is not None

        self.Up_last = Upsample(scale_factor=2)
        self.input_shortcut = nn.Sequential(
            nn.Conv3d(1, num_classes*2, 3, 1, 1),
            nn.InstanceNorm3d(num_classes*2, affine=True)
        )
        self.stackConv = StackConvNormNonlin(final_out_features+num_classes*2, num_classes*2, num_convs=2)
        self.final_conv = nn.Conv3d(num_classes*2, num_classes, 1)
        self.encoder = nn.ModuleList(self.encoder)
        self.down = nn.ModuleList(self.down)
        self.decoder = nn.ModuleList(self.decoder)
        self.up = nn.ModuleList(self.up)
        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        # print("data shape: {}".format(x.shape))
        skips = []
        input_tensor = x
        for d in range(len(self.encoder)):
            x = self.down[d](x)
            x = self.encoder[d](x)
            skips.append(x)
        skips.pop(-1)
        for u in range(len(self.decoder)):
            x = self.up[u](x)
            if skips[-(u+1)].shape != x.shape:
                # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
                pleft = abs(skips[-(u+1)].shape[-1]-x.shape[-1])//2
                ptop = abs(skips[-(u+1)].shape[-2]-x.shape[-2])//2
                pfront = abs(skips[-(u+1)].shape[-3]-x.shape[-3])//2
                x = F.pad(x, [pleft, abs(skips[-(u+1)].shape[-1]-x.shape[-1])-pleft,
                              ptop, abs(skips[-(u+1)].shape[-2]-x.shape[-2])-ptop,
                              pfront, abs(skips[-(u+1)].shape[-3]-x.shape[-3])-pfront])
            # print("{}, {}".format(skips[-(u+1)].shape, x.shape))
            x = torch.cat((torch.sigmoid(x)*skips[-(u+1)], x), dim=1)
            x = self.decoder[u](x)
        x = self.Up_last(x)
        shortcut_tensor = self.input_shortcut(input_tensor)
        if shortcut_tensor.shape != x.shape:
            # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
            pleft = abs(shortcut_tensor.shape[-1] - x.shape[-1]) // 2
            ptop = abs(shortcut_tensor.shape[-2] - x.shape[-2]) // 2
            pfront = abs(shortcut_tensor.shape[-3] - x.shape[-3]) // 2
            x = F.pad(x, [pleft, abs(shortcut_tensor.shape[-1] - x.shape[-1]) - pleft,
                          ptop, abs(shortcut_tensor.shape[-2] - x.shape[-2]) - ptop,
                          pfront, abs(shortcut_tensor.shape[-3] - x.shape[-3]) - pfront])
        # print("{}, {}".format(skips[-(u+1)].shape, x.shape))
        x = torch.cat((shortcut_tensor, x), dim=1)
        x = self.final_conv(self.stackConv(x))
        return x


if __name__ == '__main__':
    # 320, 480, 480
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    x = torch.randn(1, 1, 35, 255, 255).contiguous()
    net = GALANetv2(1, 6, x.shape[2:], [32, 64, 128]).cuda()
    print(x.shape)
    x = x.cuda()
    print(net)
    out = net.forward(x)
    print(out.shape)
