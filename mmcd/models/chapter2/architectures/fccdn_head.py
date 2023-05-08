from mmseg.core import add_prefix 
from mmseg.ops.wrappers import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
import torch
import torch.nn.functional as F


@HEADS.register_module()
class FCCDNHead(BaseDecodeHead):
    """FCCDN 

    Args:
        BaseDecodeHead (_type_): _description_
    """

    def __init__(self, **kwargs):
        super(FCCDNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
 
    def forward(self, inputs):
        """Forward function."""
        feat_t1, feat_t2 = self._transform_inputs(inputs)
        a_pred = self.cls_seg(feat_t1)
        b_pred = self.cls_seg(feat_t2)
        return (a_pred, b_pred)

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``output1`` is used."""
        return self.forward(inputs)[0]

    def _detached_sigmoid_pred(self, seg_logit):
        detached_seg_logit = seg_logit.clone()
        detached_seg_logit = F.sigmoid(detached_seg_logit)

        detached_seg_pred = (detached_seg_logit >
                             self.threshold).to(detached_seg_logit).squeeze(1)

        return detached_seg_pred

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute ``a_pred``, ``b_pred`` loss."""
        a_pred, b_pred = seg_logit
        loss = dict()
        if a_pred.shape[-1] != seg_label.shape[-1]:

            a_pred = resize(
                input=a_pred,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            b_pred = resize(
                input=b_pred,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)

        if self.out_channels != 1:
            a_pred_mask_nochange_label = a_pred.clone().argmax(1)
            a_pred_mask_change_label = a_pred.clone().argmax(1)
            b_pred_mask_nochange_label = b_pred.clone().argmax(1)
            b_pred_mask_change_label = b_pred.clone().argmax(1)
        else:
            a_pred_mask_nochange_label = self._detached_sigmoid_pred(a_pred)
            a_pred_mask_change_label = self._detached_sigmoid_pred(a_pred)
            b_pred_mask_nochange_label = self._detached_sigmoid_pred(b_pred)
            b_pred_mask_change_label = self._detached_sigmoid_pred(b_pred)

        a_pred_mask_nochange_label[seg_label == 1] = self.ignore_index
        b_pred_mask_nochange_label[seg_label == 1] = self.ignore_index

        a_pred_mask_change_label = seg_label - a_pred_mask_change_label  # 会产生负数，下一步消除负数区域
        a_pred_mask_change_label[seg_label == 0] = self.ignore_index
        b_pred_mask_change_label = seg_label - b_pred_mask_change_label
        b_pred_mask_change_label[seg_label == 0] = self.ignore_index

        # print(a_pred_mask_change_label.shape)

        # 可以跑几个iter
        # bb = seg_label ^ a_pred_mask_change_label.clone() #
        # a_pred_mask_change_label[bb] = 1
        # a_pred_mask_change_label[seg_label == 0] = self.ignore_index

        # aa = seg_label ^ b_pred_mask_change_label.clone()
        # b_pred_mask_change_label[aa] = 1
        # b_pred_mask_change_label[seg_label == 0] = self.ignore_index

        loss.update(
            add_prefix(
                super(FCCDNHead, self).losses(
                    a_pred, b_pred_mask_nochange_label.to(torch.long)),
                'nochange_a2b'))
        loss.update(
            add_prefix(
                super(FCCDNHead, self).losses(
                    b_pred, a_pred_mask_nochange_label.to(torch.long)),
                'nochange_b2a'))

        # ---bug  acc can not calculate in cuda 
        # cause: seglabe has 0,254,255 other class 
        # RuntimeError: CUDA error: an illegal memory access was encountered
        #  fixed: dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0), # seg_pad_val注意！！！！ fccdn 需要0 填充 默认255
         

        loss.update(
            add_prefix(
                super(FCCDNHead, self).losses(
                    a_pred, b_pred_mask_change_label.to(torch.long)),
                'change_a2b'))
        loss.update(
            add_prefix(
                super(FCCDNHead, self).losses(
                    b_pred, a_pred_mask_change_label.to(torch.long)),
                'change_b2a'))

        return loss

