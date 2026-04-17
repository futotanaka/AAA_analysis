import torch
import torch.nn as nn

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # print('g:',g.shape,'x:',x.shape)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

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

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, first_filter_num=64):
        super(AttentionUNet, self).__init__()

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

        # self.up_conv4 = UpConv(16 * first_filter_num, 8 * first_filter_num, 3)
        self.up_conv4 = UpConv(16 * first_filter_num, 8 * first_filter_num, 3)
        self.Att4 = Attention_block(8 * first_filter_num, 8 * first_filter_num, 4 * first_filter_num)
        self.double_conv4 = DoubleConv(16 * first_filter_num, 8 * first_filter_num, 3)

        self.up_conv3 = UpConv(8 * first_filter_num, 4 * first_filter_num, 3)
        self.Att3 = Attention_block(4 * first_filter_num, 4 * first_filter_num, 2 * first_filter_num)
        self.double_conv3 = DoubleConv(8 * first_filter_num, 4 * first_filter_num, 3)

        self.up_conv2 = UpConv(4 * first_filter_num, 2 * first_filter_num, 3)
        self.Att2 = Attention_block(2 * first_filter_num, 2 * first_filter_num, 1 * first_filter_num)
        self.double_conv2 = DoubleConv(4 * first_filter_num, 2 * first_filter_num, 3)

        self.up_conv1 = UpConv(2 * first_filter_num, 1 * first_filter_num, 3)
        self.Att1 = Attention_block(1 * first_filter_num, 1 * first_filter_num, first_filter_num // 2)
        self.double_conv1 = DoubleConv(2 * first_filter_num, 1 * first_filter_num, 3)

        self.conv_final = nn.Conv2d(first_filter_num, out_channels, 1)

    def forward(self, x):

        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)

        d5 = self.up_conv4(x5)
        x4 = self.Att4(d5,x4)
        d5 = torch.cat((x4,d5), dim=1)
        d5 = self.double_conv4(d5)
        # print('d5:',d5.shape,'x4:',x4.shape)

        d4 = self.up_conv3(d5)
        # print('d4:',d4.shape,'x3:',x3.shape)
        x3 = self.Att3(d4,x3)
        d4 = torch.cat((x3,d4), dim=1)
        d4 = self.double_conv3(d4)

        d3 = self.up_conv2(d4)
        x3 = self.Att2(d3,x2)
        d3 = torch.cat((x2,d3), dim=1)
        d3 = self.double_conv2(d3)
        
        d2 = self.up_conv1(d3)
        x1 = self.Att1(d2,x1)
        d2 = torch.cat((x1,d2), dim=1)
        d2 = self.double_conv1(d2)

        return torch.sigmoid(self.conv_final(d2))
        # return self.conv_final(x)
