"""
Created on Sep 21 2023
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
            nn.Conv3d(in_channels, out_channels,
                      kernel_size, padding=pad_size),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels,
                      kernel_size, padding=pad_size),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        pad_size = kernel_size // 2
        self.is_ch_changed = (in_channels != out_channels)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=pad_size)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=pad_size)
        if self.is_ch_changed:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1, padding=0)
    def forward(self, x):
        shortcut = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.is_ch_changed:
            shortcut = self.shortcut(x)
        out += shortcut
        return out
class ConvUnit3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 use_residual_block=False):
        super(ConvUnit3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual_block = use_residual_block
        if self.use_residual_block:
            self.conv = ResidualBlock(self.in_channels,
                                      self.out_channels,
                                      kernel_size)
        else:
            self.conv = DoubleConv(self.in_channels,
                                   self.out_channels,
                                   kernel_size)
    def forward(self, x):
        x = self.conv(x)
        return x
class UnetPP3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=5, first_filter_num=64,
                 use_residual_block=False):
        super(UnetPP3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_filter_num = first_filter_num
        self.depth = depth
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2)
        self.conv_0_0 = ConvUnit3D(in_channels, first_filter_num, 3, use_residual_block)
        self.conv_1_0 = ConvUnit3D(first_filter_num, 2 * first_filter_num, 3, use_residual_block)
        self.conv_2_0 = ConvUnit3D(2 * first_filter_num, 4 * first_filter_num, 3, use_residual_block)
        self.conv_1_1 = ConvUnit3D(6 * first_filter_num, 2 * first_filter_num, 3, use_residual_block)
        self.conv_0_1 = ConvUnit3D(3 * first_filter_num, first_filter_num, 3, use_residual_block)
        self.conv_0_2 = ConvUnit3D(4 * first_filter_num, first_filter_num, 3, use_residual_block)
        if self.depth >= 4:
            self.conv_3_0 = ConvUnit3D(4 * first_filter_num, 8 * first_filter_num, 3, use_residual_block)
            self.conv_2_1 = ConvUnit3D(12 * first_filter_num, 4 * first_filter_num, 3, use_residual_block)
            self.conv_1_2 = ConvUnit3D(8 * first_filter_num, 2 * first_filter_num, 3, use_residual_block)
            self.conv_0_3 = ConvUnit3D(5 * first_filter_num, first_filter_num, 3, use_residual_block)
        if self.depth == 5:
            self.conv_4_0 = ConvUnit3D(8 * first_filter_num, 16 * first_filter_num, 3, use_residual_block)
            self.conv_3_1 = ConvUnit3D(24 * first_filter_num, 8 * first_filter_num, 3, use_residual_block)
            self.conv_2_2 = ConvUnit3D(16 * first_filter_num, 4 * first_filter_num, 3, use_residual_block)
            self.conv_1_3 = ConvUnit3D(10 * first_filter_num, 2 * first_filter_num, 3, use_residual_block)
            self.conv_0_4 = ConvUnit3D(6 * first_filter_num, first_filter_num, 3, use_residual_block)
        self.conv_final = nn.Conv3d(first_filter_num, out_channels, 1)
    def forward(self, x):
        out_0_0 = self.conv_0_0(x)
        out_1_0 = self.conv_1_0(self.pool(out_0_0))
        out_2_0 = self.conv_2_0(self.pool(out_1_0))
        out_1_1 = self.conv_1_1(torch.cat((out_1_0, self.up(out_2_0)), 1))
        out_0_1 = self.conv_0_1(torch.cat([out_0_0, self.up(out_1_0)], 1))
        out_0_2 = self.conv_0_2(torch.cat([out_0_0, out_0_1, self.up(out_1_1)], 1))
        if self.depth >= 4:
            out_3_0 = self.conv_3_0(self.pool(out_2_0))
            out_2_1 = self.conv_2_1(torch.cat([out_2_0, self.up(out_3_0)], 1))
            out_1_2 = self.conv_1_2(torch.cat([out_1_0, out_1_1, self.up(out_2_1)], 1))
            out_0_3 = self.conv_0_3(torch.cat([out_0_0, out_0_1, out_0_2, self.up(out_1_2)], 1))
        if self.depth == 5:
            out_4_0 = self.conv_4_0(self.pool(out_3_0))
            out_3_1 = self.conv_3_1(torch.cat([out_3_0, self.up(out_4_0)], 1))
            out_2_2 = self.conv_2_2(torch.cat([out_2_0, out_2_1, self.up(out_3_1)], 1))
            out_1_3 = self.conv_1_3(torch.cat([out_1_0, out_1_1, out_1_2, self.up(out_2_2)], 1))
            final_in= self.conv_0_4(torch.cat([out_0_0, out_0_1, out_0_2, out_0_3, self.up(out_1_3)], 1))
        elif self.depth == 4:
            final_in = out_0_3
        elif self.depth == 3:
            final_in = out_0_2
        return torch.sigmoid(self.conv_final(final_in))