import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import FUSE




@FUSE.register_module()
class FPNFuse(BaseModule):
    def __init__(self,
        in_channels,
        mid_channel_3x3,
        out_channel_3x3,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.channels = mid_channel_3x3
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # -----------------------
        # 相比 segformer 在 7张图象上表现很好
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            fpn_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
        
        self.fpn_bottleneck = nn.Sequential(
            ConvModule(
                len(self.in_channels) * self.channels,
                out_channel_3x3,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Dropout(0.5),
            ConvModule(
                out_channel_3x3,
                out_channel_3x3,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        # -----------------------
    
    def forward(self,inputs):
        fpn_outs = [
            self.fpn_convs[i](inputs[i])
            for i in range(len(self.in_channels))
        ]

        for i in range(len(self.in_channels)):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        out = self.fpn_bottleneck(fpn_outs)
        return out