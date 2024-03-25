# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/25 23:15
@Auth ： YongTong Gu
@File ：sparse_spatial_pooling.py
@IDE ：PyCharm
@Motto：悟已往之不谏,知来者之可追

"""
from torch import nn
import torch
import torch.nn.functional as F


class SSP(nn.Module):
    def __init__(self, in_channel):
        depth = in_channel
        super(SSP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net


if __name__ == '__main__':
    ssp = SSP(256)
    input = torch.rand(2, 256, 13, 13)
    output = ssp(input)
    print(output.shape)
