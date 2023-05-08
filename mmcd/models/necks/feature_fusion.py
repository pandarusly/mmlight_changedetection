import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16
# from .unet_neck import ResUNetDecoder,ResUNetPixelShuffleDecoder


@NECKS.register_module()
class FeatureFusionNeck(BaseModule):
    """Feature Fusion Neck.

    Args:
        policy (str): The operation to fuse features. candidates 
            are `concat`, `sum`, `diff` and `Lp_distance`.
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        out_indices (tuple[int]): Output from which layer.
    """

    def __init__(self,
                  policy="Lp_distance",
                 in_channels=None,
                 channels=None,
                 out_indices=(0, 1, 2, 3)):
        super(FeatureFusionNeck, self).__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices
        self.fp16_enabled = False

    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""
        
        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
            'supported'.format(_fusion_policies)
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'Lp_distance':
            x = torch.abs(x1 - x2)

        return x

    @auto_fp16()
    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"
        outs = []
        for i in range(len(x1)):
            out = self.fusion(x1[i], x2[i], self.policy)
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
    

@NECKS.register_module()
class UnetFeatureFusionNeck(BaseModule):

    def __init__(self, 
                 in_channels,
                 channels,
                 policy='concat',
                 pixel_shuff=None, 
                 out_indices=(0, 1, 2, 3)):
        super(UnetFeatureFusionNeck, self).__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices
        self.fp16_enabled = False

        if pixel_shuff:
            self.unet = ResUNetDecoder(in_channels,channels)
        else:
            self.unet = ResUNetPixelShuffleDecoder(in_channels,channels)


    @staticmethod
    def fusion(x1, x2, policy="Lp_distance"):
        """Specify the form of feature fusion"""
        
        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
            'supported'.format(_fusion_policies)
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'Lp_distance':
            x = torch.abs(x1 - x2)
 

        return x

    @auto_fp16()
    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"
        
        # ----
        x1 = self.unet(x1)
        x2 = self.unet(x2)

        outs = []
        for i in range(len(x1)):
            out = self.fusion(x1[i], x2[i], self.policy)
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
 

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


@NECKS.register_module()
class ResUNetDecoder(BaseModule):
    def __init__(self, input_channels, mid_channels):
        super(ResUNetDecoder, self).__init__()
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

        outs = [resnet_outputs[0]]
        
        for i in range(self.num_layers): 
            if x.shape[2:] != resnet_outputs[i + 1].shape[2:]:
                x = F.interpolate(x, size=resnet_outputs[i + 1].shape[2:], mode='bilinear', align_corners=False)
            x = self.layers[2 * i](x)
            x = torch.cat([x, resnet_outputs[i + 1]], dim=1)
            x = self.layers[2 * i + 1](x)
            outs.append(x)
        # x = self.final_conv(x)
        # return x
        return tuple(reversed(outs))


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(PixelShuffleUpsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    

@NECKS.register_module()
class ResUNetPixelShuffleDecoder(BaseModule):
    def __init__(self, input_channels, mid_channels):
        super(ResUNetPixelShuffleDecoder, self).__init__()
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
        outs = [resnet_outputs[0]]
        for i in range(self.num_layers):
            # print(x.shape)
            if x.shape[2:] != resnet_outputs[i + 1].shape[2:]:
                x = self.upsample_layers[i](x) 
            # print(x.shape)
            x = self.layers[2 * i](x)
            x = torch.cat([x, resnet_outputs[i + 1]], dim=1)
            x = self.layers[2 * i + 1](x)
            outs.append(x)
        # x = self.final_conv(x)
        return tuple(reversed(outs))