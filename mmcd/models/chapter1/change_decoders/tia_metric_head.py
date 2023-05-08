import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ..builder import build_attention,build_fuse


@HEADS.register_module()
class TIAMetricHead(BaseDecodeHead):
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
        distance_threshold=1, 
        **kwargs):
        super().__init__(input_transform='multiple_select', num_classes=1, **kwargs)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.distance_threshold = distance_threshold
        # -----------------------
        # FPN 相比 segformer 在 7张图象上表现很好 in Metric
        self.fuse_conv = build_fuse(fuse_cfg)
        # -----------------------
        
        self.netA = build_attention(attn) if attn else None

        self.calc_dist = nn.PairwiseDistance(keepdim=True)

        self.conv_seg = nn.Identity()
                
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
        if self.netA:
            f1, f2 = self.netA(f1, f2)

        # if you use PyTorch<=1.9, there may be some problems. 
        # see https://github.com/justchenhao/STANet/issues/85
        # f1 = f1.permute(0, 2, 3, 1)
        # f2 = f2.permute(0, 2, 3, 1)
        # dist = self.calc_dist(f1, f2).permute(0, 3, 1, 2)

        dist = self.calc_dist(f1, f2)

        dist = F.interpolate(dist, size=inputs[0].shape[2:], mode='bilinear', align_corners=True)

        return dist
    
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
        # adapt `sigmoid` in post-processing
        seg_logits[seg_logits > self.distance_threshold] = 100
        seg_logits[seg_logits <= self.distance_threshold] = -100
        return seg_logits


 