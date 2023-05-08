import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import FUSE

@FUSE.register_module()
class SegformerFuse(BaseModule):
    def __init__(self,
        in_channels,
        mid_channel_1x1,
        out_channel_1x1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.channels = mid_channel_1x1
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        num_inputs = len(in_channels)
        # -----------------------
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=out_channel_1x1,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # -----------------------
    
    def forward(self,inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        
        return out