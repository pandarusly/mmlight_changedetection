from mmcd.models.change_detectors import SiamEncoderDecoder
from mmseg.models.builder import SEGMENTORS
import torch


@SEGMENTORS.register_module()
class SemiSiamEncoderDecoder(SiamEncoderDecoder):
    """_summary_
    train_step -> data_batch dict_keys(['sup_data', 'unsup_data'])
    

    Args:
         
    """
    def __init__(self, backbone, decode_head, neck=None,auxiliary_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None, backbone_inchannels=3):
        super().__init__(backbone, decode_head, neck, auxiliary_head, train_cfg, test_cfg, pretrained, init_cfg, backbone_inchannels)
 

    

    def train_step(self, data_batch, optimizer, **kwargs):
        
        # print(data_batch.keys()) dict_keys(['sup_data', 'unsup_data'])
        # if not data_batch.has_key('unsup_data'):
        if 'unsup_data' not in data_batch.keys(): 
            data_batch['unsup_data'] = None

        losses = self.forward_train(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['sup_data']['img_metas'])
            )

        return outputs


    def forward_train(self, sup_data, unsup_data):
        
        # ------------------ super 
        sup_img, sup_img_metas, gt_semantic_seg = sup_data['img'],sup_data['img_metas'],sup_data['gt_semantic_seg']

        sup_x = self.extract_feat(sup_img)
        losses = dict()
        loss_decode_sup = self._decode_head_forward_train(sup_x,  sup_img_metas, gt_semantic_seg)
        
        losses.update(loss_decode_sup)
        if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                sup_x, sup_img_metas, gt_semantic_seg)
                losses.update(loss_aux)
        # ------------------ un_super 
        if unsup_data is not None:
            unsup_img, unsup_img_metas, _ = unsup_data['img'],unsup_data['img_metas'],None 

            unsup_x = self.extract_feat(unsup_img)

            # gt_semantic_seg = None ,注意力loss名字 不要重复，放在 decode里面高
            loss_decode_unsup = self._decode_head_forward_train(unsup_x, unsup_img_metas,
                                                        None)
            losses.update(loss_decode_unsup)

        # print(losses)
        return losses
   

 
