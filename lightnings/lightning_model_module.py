from abc import ABC
from collections import OrderedDict
from typing import Any, Dict

import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from omegaconf import OmegaConf
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.module import LightningModule
from lightnings.utils.binary import ConfuseMatrixMeter

from mmseg.core import add_prefix
from mmseg.models import build_segmentor
from .utils import build_optimizer, build_scheduler


class BaseModelModule(LightningModule, ABC):
    def __init__(
            self,
            cfg,  # mmcv
            Config,  # OmegaConf
            evaluate_dataset,
            eval_kwargs,
            HPARAMS_LOG=False,
            CKPT=False,
            **kwargs
    ):
        super().__init__()

        Config = OmegaConf.create(Config)
        self.save_hyperparameters(Config, logger=HPARAMS_LOG)
        self.lr = self.hparams.TRAIN.BASE_LR
        model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
        model.CLASSES = cfg.CLASSES
        model.ignore_index = cfg.ignore_index
        
        self.model = model
        self.model.init_weights()
        self.model = revert_sync_batchnorm(self.model)

        self.CLASSES = model.CLASSES
        self.PALETTE = model.PALETTE
        self.evaluate_dataset = evaluate_dataset
        self.eval_kwargs = eval_kwargs

        # example_input_array=(1, 6, 256, 256)
        # self.example_input_array = torch.randn(*example_input_array)
  

        if CKPT:
            self._finetue(CKPT)

    def _finetue(self, ckpt_path):
        print("-" * 30)
        print("locate new momdel pretrained {}".format(ckpt_path))
        print("-" * 30)
        pretained_dict = torch.load(ckpt_path)["state_dict"]
        self.load_state_dict(pretained_dict)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.model.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.model.forward_train(img, img_metas, **kwargs)
        else:
            return self.model.forward_test(img, img_metas, **kwargs)

    def forward_dummy(self, img):
        """
        为了计算flops
        """
        return self.model.forward_dummy(img)
    
    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams, self.parameters())
        scheduler = build_scheduler(self.hparams, optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.TRAIN.INTERVAL,
                "monitor": self.hparams.TRAIN.MONITOR
            },
        }

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     # print("\ncurrent_epoch ", self.current_epoch)
    #     # print("global_step", self.global_step)
    #     scheduler.step(
    #         epoch=self.current_epoch
    #     )  # timm's scheduler need the epoch value

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """_summary_

        Args:
            batch (Dict[str, Any]): train dataloder 的输出
            batch_idx (int): 占位

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """

        losses = self(**batch)
        loss, log_vars = self.model._parse_losses(losses)
        log_vars.pop('loss') # lighting自带追踪
        # ------------------
        # outputs = {
        #     'loss': 0.5,
        #     'log_vars': {
        #         'accuracy': 0.98
        #     },
        # }
        # ------------------
        self.log_dict(log_vars, prog_bar=True, sync_dist=True,on_step=False,on_epoch=True)
        return {"loss": loss}

      

class BaseModelModuleV2(BaseModelModule):

    def __init__(self, cfg, Config, evaluate_dataset, eval_kwargs, HPARAMS_LOG=False, CKPT=False, **kwargs):
        super().__init__(cfg, Config, evaluate_dataset, eval_kwargs, HPARAMS_LOG, CKPT, **kwargs)
    
        num_classes = len(self.CLASSES)
        metric = self.eval_kwargs.get('metric',["mIoU",])
        self.metrics=["mIoU","mFscore"]
        metric=["mIoU","mFscore"]
        self.train_metrics = ConfuseMatrixMeter(num_classes,metric)
        self.val_metrics = ConfuseMatrixMeter(num_classes,metric)
        self.test_metrics = ConfuseMatrixMeter(num_classes,metric)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        imgs = batch['img'] 
        gt_semantic_seg = batch['gt_semantic_seg']
        seg_logit = self.forward_dummy(imgs)
        if self.model.out_channels == 1:
            seg_pred = (seg_logit >
                        self.model.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)

        gt_semantic_seg = gt_semantic_seg.squeeze(1)

        self.val_metrics(seg_pred, gt_semantic_seg)

    def validation_epoch_end(self, outputs: Any) -> None:
        """ 验证集可以总体算混淆举证
        """
        metrics = self.val_metrics.compute()
        for metric in self.metrics:
            res = metrics.get(metric)
            # print(metric,res)
            for k,v in  zip(self.CLASSES,res):
                metrics.update({metric+'.'+k: v})
            metrics.update({metric: res.mean()})
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        imgs = batch['img']
        gt_semantic_seg = batch['gt_semantic_seg']
        seg_logit =  self.forward_dummy(imgs)
        if self.model.out_channels == 1:
            seg_pred = (seg_logit >
                        self.model.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)

        gt_semantic_seg = gt_semantic_seg.squeeze(1)
        self.test_metrics(seg_pred, gt_semantic_seg)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        metrics = self.test_metrics.compute()
        for metric in self.metrics:
            res = metrics.get(metric)
            for k,v in  zip(self.CLASSES,res):
                metrics.update({metric+'.'+k: v})
            metrics.update({metric: res.mean()})
        log_vars = add_prefix(metrics, "test")
        self.log_dict(log_vars, prog_bar=True)
    


    def on_save_checkpoint(self, checkpoint):
 
        new_ckpt = OrderedDict() 
        ckpt =  checkpoint['state_dict'] 
        checkpoint.clear()

        for k, v in ckpt.items(): 
            if k.startswith('model'):
                new_k = k.replace('model.', '')
                new_v = v
            new_ckpt[new_k] = new_v 

        checkpoint['state_dict'] =  new_ckpt
        meta = dict(
            mmseg_version='00',
            CLASSES=self.CLASSES,
            PALETTE=self.PALETTE,
            )
        checkpoint['meta'] =  meta
 