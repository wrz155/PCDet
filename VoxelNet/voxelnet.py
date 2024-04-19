import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# fully connected network
class FCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN, self).__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t = x.shape
        x = self.linear(x.view(kk*t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)

# Voxel Feature Encoding layer
class VFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFE, self).__init__()
        assert out_channels % 2 == 0
        self.units = out_channels // 2
        self.fcn = FCN(in_channels, out_channels)

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):


# Convolutional Middle Layer
class CML(nn.Module):

# Region Proposal Network
class RPN(nn.Module):