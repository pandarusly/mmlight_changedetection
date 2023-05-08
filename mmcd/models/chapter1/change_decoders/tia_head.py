import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ..builder import build_attention,build_difference,build_fuse
 


@HEADS.register_module()
class TIAHead(BaseDecodeHead):
    """The Head of TIANet for Metric.

    Args:
        distance_threshold=1,
        attn=True,
        interpolate_mode='bilinear',
    """

    def __init__(
        self, 
        attn=None,
        difference_cfg=dict(type='abs_diff'),
        fuse_cfg=dict(type='FPNFuse',norm_cfg = dict(type='BN', requires_grad=True)),
        **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        # -----------------------
        # FPN 相比 segformer 在 7张图象上表现很好 in Metric
        self.fuse_conv = build_fuse(fuse_cfg)
        # -----------------------
        
        self.netA = build_attention(attn) if attn else None

        self.calc_dist = build_difference(difference_cfg)

        # self.channels  == self.calc_dist.out_channel?
        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

                
    def base_forward(self, inputs):
        feats = self.fuse_conv(inputs)
        return feats

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        
        f1 = self.base_forward(inputs1)
        f2 = self.base_forward(inputs2)
        # if self.netA:
        #     f1, f2 = self.netA(f1, f2)

        # if you use PyTorch<=1.9, there may be some problems. 
        # see https://github.com/justchenhao/STANet/issues/85
        # f1 = f1.permute(0, 2, 3, 1)
        # f2 = f2.permute(0, 2, 3, 1)
        # dist = self.calc_dist(f1, f2).permute(0, 3, 1, 2)

        out = self.calc_dist(f1, f2)

        out = self.cls_seg(out)

        return out
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
    
        return losses
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs)
        return seg_logits

@HEADS.register_module()
class TIAChangeUnetHead(BaseDecodeHead):
    def __init__(self, attn=None, difference_cfg=[dict(type='abs_diff')], fuse_cfg=dict(type='ResUNetFuse',
            input_channels=list(reversed([32, 64, 160, 256])),
            mid_channels=64,), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.fuse_conv = build_fuse(fuse_cfg)
      
        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

        assert isinstance(difference_cfg,list)
        self.calc_dist = nn.ModuleList(
            [
            build_attention(difference_) for difference_ in difference_cfg
            ]
        ) 

    def base_forward(self, inputs,inputEx,):
        pass

    
    @property
    def with_attn(self):
        return hasattr(self, 'netA') and self.netA is not None

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = [] # [256,64,64,64] 1/32 -> 
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)

        cc = []
        for id, calc_dist in  enumerate(self.calc_dist):
            cc.append(
                calc_dist(inputs1[id],inputs2[id])
            )
 
        cc = self.fuse_conv(cc)
        # print(f"cc shape: {cc.shape} ")
        cc = self.cls_seg(cc)

        return cc