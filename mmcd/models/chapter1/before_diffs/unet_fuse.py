import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import FUSE


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x): 
        x = self.conv1x1(x) 
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) 
        x += residual
        x = self.relu(x)
        return x


@FUSE.register_module()
class ResUNetFuse(BaseModule):
    def __init__(self, input_channels, mid_channels):
        super(ResUNetFuse, self).__init__()
        assert len(input_channels) > 1, "Input channels list should have at least two elements."
        assert mid_channels > 0, "Mid channels should be a positive integer."

        self.layers = nn.ModuleList()
        self.num_layers = len(input_channels) - 1
        for i in range(self.num_layers):
            if i == 0:
                input_channel = input_channels[i]
            else:
                input_channel = mid_channels
            self.layers.append(nn.Conv2d(input_channel, mid_channels, kernel_size=1))
            self.layers.append(ResidualBlock(input_channels[i + 1] + mid_channels, mid_channels))
        # self.final_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, resnet_outputs):
        resnet_outputs = list(reversed(resnet_outputs))
        x = resnet_outputs[0]

        
        for i in range(self.num_layers): 
            if x.shape[2:] != resnet_outputs[i + 1].shape[2:]:
                x = F.interpolate(x, size=resnet_outputs[i + 1].shape[2:], mode='bilinear', align_corners=False)
            x = self.layers[2 * i](x)
            x = torch.cat([x, resnet_outputs[i + 1]], dim=1)
            x = self.layers[2 * i + 1](x)
        # x = self.final_conv(x)
        return x


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(PixelShuffleUpsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    

@FUSE.register_module()
class ResUNetPixelShuffleFuse(BaseModule):
    def __init__(self, input_channels, mid_channels):
        super(ResUNetPixelShuffleFuse, self).__init__()
        assert len(input_channels) > 1, "Input channels list should have at least two elements."
        assert mid_channels > 0, "Mid channels should be a positive integer."

        self.layers = nn.ModuleList()
        self.num_layers = len(input_channels) - 1
        self.upsample_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                input_channel = input_channels[i]
            else:
                input_channel = mid_channels
            self.upsample_layers.append(PixelShuffleUpsample(input_channel, input_channel,2))
            self.layers.append(nn.Conv2d(input_channel, mid_channels, kernel_size=1))
            self.layers.append(ResidualBlock(input_channels[i + 1] + mid_channels, mid_channels))

        # self.final_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, resnet_outputs):
        resnet_outputs = list(reversed(resnet_outputs))
        x = resnet_outputs[0]
        for i in range(self.num_layers):
            # print(x.shape)
            if x.shape[2:] != resnet_outputs[i + 1].shape[2:]:
                x = self.upsample_layers[i](x) 
            # print(x.shape)
            x = self.layers[2 * i](x)
            x = torch.cat([x, resnet_outputs[i + 1]], dim=1)
            x = self.layers[2 * i + 1](x)
        # x = self.final_conv(x)
        return x 