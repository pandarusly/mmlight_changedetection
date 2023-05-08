import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
import torch 

from mmseg.core import add_prefix 
from mmseg.ops import resize
from mmseg.models.builder import HEADS,build_loss

from mmcd.models.chapter1.change_decoders import TIAHead


@HEADS.register_module()
class FccdnTIAHead(TIAHead):
    def __init__(self, loss_unsup_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.2),attn=None, difference_cfg=..., fuse_cfg=..., **kwargs):
        super().__init__(attn, difference_cfg, fuse_cfg, **kwargs)

        if isinstance(loss_unsup_decode, dict):
            self.loss_unsup_decode = build_loss(loss_unsup_decode)
        elif isinstance(loss_unsup_decode, (list, tuple)):
            self.loss_unsup_decode = nn.ModuleList()
            for loss in loss_unsup_decode:
                self.loss_unsup_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_unsup_decode must be a dict or sequence of dict,\
                but got {type(loss_unsup_decode)}')


                
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

        out = self.calc_dist(f1, f2)

        out = self.cls_seg(out)

        return (out , f1, f2) 
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor or None): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits,f1, f2 = self.forward(inputs)
        if gt_semantic_seg is not None:
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            gt_semantic_seg = seg_logits.detach()
            if self.out_channels != 1:
                gt_semantic_seg = gt_semantic_seg.argmax(1)
            else:
                gt_semantic_seg = F.sigmoid(gt_semantic_seg)
                gt_semantic_seg = (gt_semantic_seg >
                                    self.threshold).to(gt_semantic_seg).squeeze(1)
            
            f1 = self.cls_seg(f1) #  保证最后的分类器对于建筑物语义信息同样具有判别能力
            f2 = self.cls_seg(f2)
            losses = self.unsup_losses_all((f1, f2), gt_semantic_seg)  
    
        return losses


    def forward_test(self, inputs, img_metas, test_cfg):
        seg_logits , _, _  = self.forward(inputs)
        return seg_logits
        
    def unsup_losses_all(self, seg_logit, seg_label):
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
                self.unsup_losses(
                    a_pred, b_pred_mask_nochange_label.to(torch.long)),
                'nochange_a2b'))
        loss.update(
            add_prefix(
                self.unsup_losses(
                    b_pred, a_pred_mask_nochange_label.to(torch.long)),
                'nochange_b2a'))

        # ---bug  acc can not calculate in cuda 
        # cause: seglabe has 0,254,255 other class 
        # RuntimeError: CUDA error: an illegal memory access was encountered
        #  fixed: dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0), # seg_pad_val注意！！！！ fccdn 需要0 填充 默认255
         

        loss.update(
            add_prefix(
                self.unsup_losses(
                    a_pred, b_pred_mask_change_label.to(torch.long)),
                'change_a2b'))
        loss.update(
            add_prefix(
                self.unsup_losses(
                    b_pred, a_pred_mask_change_label.to(torch.long)),
                'change_b2a'))

        return loss



    @force_fp32(apply_to=('seg_logit', ))
    def unsup_losses(self, seg_logit, seg_label):
        assert seg_logit.requires_grad == True and seg_label.requires_grad == False
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)


        if not isinstance(self.loss_unsup_decode, nn.ModuleList):
            losses_decode = [self.loss_unsup_decode]
        else:
            losses_decode = self.loss_unsup_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        return loss

    def _detached_sigmoid_pred(self, seg_logit):
        detached_seg_logit = seg_logit.clone()
        detached_seg_logit = F.sigmoid(detached_seg_logit)

        detached_seg_pred = (detached_seg_logit >
                             self.threshold).to(detached_seg_logit).squeeze(1)

        return detached_seg_pred


@HEADS.register_module()
class FccdnTIAHeadV2(FccdnTIAHead):
    def __init__(self, loss_unsup_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2), attn=None, difference_cfg=..., fuse_cfg=..., **kwargs):
        super().__init__(loss_unsup_decode, attn, difference_cfg, fuse_cfg, **kwargs)

        # layers = []

        # # Middle channels
        # mid_channels = 32

        # #First conv layer to reduce number of channels
        # diff_conv1x1 = nn.Conv2d(self.channels, mid_channels, kernel_size=3, padding=1)
        # nn.init.kaiming_normal_(diff_conv1x1.weight.data, nonlinearity='relu')
        # layers.append(diff_conv1x1)

        # #ReLU
        # diff_relu = nn.ReLU()
        # layers.append(diff_relu)

 
        # #Classification layer
        # conv1x1 = nn.Conv2d(mid_channels, self.out_channels, kernel_size=1)
        # nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
        # layers.append(conv1x1)
        
        # self.unsup_classify = nn.Sequential(
        #     *layers
        # )
        self.unsup_classify = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits,f1, f2 = self.forward(inputs)
        if gt_semantic_seg is not None:
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            gt_semantic_seg = seg_logits.detach()
            if self.out_channels != 1:
                gt_semantic_seg = gt_semantic_seg.argmax(1)
            else:
                gt_semantic_seg = F.sigmoid(gt_semantic_seg)
                gt_semantic_seg = (gt_semantic_seg >
                                    self.threshold).to(gt_semantic_seg).squeeze(1)
            
            f1 = self.unsup_classify(f1) #  保证最后的分类器对于建筑物语义信息同样具有判别能力
            f2 = self.unsup_classify(f2)
            losses = self.unsup_losses_all((f1, f2), gt_semantic_seg)  
    
        return losses