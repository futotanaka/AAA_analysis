"""
Created on Sun Oct 23 2022
@author: ynomura
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    # (conv => BatchNorm => ReLU) x 2
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DoubleConv, self).__init__()
        pad_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=pad_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, padding=pad_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv = DoubleConv(self.in_channels,
                               self.out_channels,
                               kernel_size)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = DoubleConv(self.in_channels,
                               self.out_channels,
                               kernel_size)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upsample(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, first_filter_num=64):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_filter_num = first_filter_num

        self.down_conv1 = DownConv(in_channels, first_filter_num, 3, False)
        self.down_conv2 = DownConv(first_filter_num, 2 * first_filter_num, 3)
        self.down_conv3 = DownConv(
            2 * first_filter_num, 4 * first_filter_num, 3)
        self.down_conv4 = DownConv(
            4 * first_filter_num, 8 * first_filter_num, 3)
        self.down_conv5 = DownConv(
            8 * first_filter_num, 16 * first_filter_num, 3)

        self.up_conv4 = UpConv(24 * first_filter_num, 8 * first_filter_num, 3)
        self.up_conv3 = UpConv(12 * first_filter_num, 4 * first_filter_num, 3)
        self.up_conv2 = UpConv(6 * first_filter_num, 2 * first_filter_num, 3)
        self.up_conv1 = UpConv(3 * first_filter_num, first_filter_num, 3)

        self.conv_final = nn.Conv2d(first_filter_num, out_channels, 1)

    def forward(self, x):

        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)
        x = self.up_conv4(x4, x5)
        x = self.up_conv3(x3, x)
        x = self.up_conv2(x2, x)
        x = self.up_conv1(x1, x)

        return torch.sigmoid(self.conv_final(x))
        # return self.conv_final(x)
