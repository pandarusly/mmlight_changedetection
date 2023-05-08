from abc import ABC
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from mmseg.datasets import build_dataset
from mmseg.datasets.builder import build_dataloader


class BaseDCDataModule(pl.LightningDataModule, ABC):
    def __init__(self, cfg) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams.cfg' attribute
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # The default loader config
        loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=1,
            dist=False,
            seed=None,
            drop_last=True)
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })

        self.loader_cfg = loader_cfg

    def setup(self, stage: Optional[str] = None):
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:

            self.train_dataset = build_dataset(self.hparams.cfg.train)
            self.val_dataset = build_dataset(self.hparams.cfg.val)
            self.test_dataset = build_dataset(self.hparams.cfg.test)

    def train_dataloader(self) -> DataLoader:
        # The specific dataloader settings
        train_loader_cfg = {**self.loader_cfg, **
                            self.hparams.cfg.get('train_dataloader', {})}
        return build_dataloader(self.train_dataset, **train_loader_cfg)

    def val_dataloader(self) -> DataLoader:
        # The specific dataloader settings
        val_loader_cfg = {
            **self.loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **self.hparams.cfg.get('val_dataloader', {}),
        }
        return build_dataloader(self.val_dataset, **val_loader_cfg)

    def test_dataloader(self) -> DataLoader:

        test_loader_cfg = {
            **self.loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **self.hparams.cfg.get('test_dataloader', {})
        }
        return build_dataloader(self.test_dataset, **test_loader_cfg)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # if batch.get('gt_semantic_seg'):
        #     batch["gt_semantic_seg"] = batch["gt_semantic_seg"].data
        # batch["img"] = batch["img"]
        # batch["img_metas"] = batch["img_metas"].data
        # print(len(batch['img_metas']))
        out = dict()
        if not self.trainer.training:
            for k, v in batch.items():
                # print(k,v)
                new_v = self.DC2Tensor(v)
                # print(k,new_v)
                out.update({k: new_v})
        else:
            for k, v in batch.items():
                new_v = self.DC2Tensor(v)
                new_v = new_v[0] if k == 'img_metas' else new_v[0]
                out.update({k: new_v})
        return out

    def DC2Tensor(self, DataContainer):
        """ DataContainer in mmseg has three keys, and is cpu_only. it is not useful to Lightning Dataset
        batch['img_metas'] 
        batch['img']
        batch['gt_semantic_seg']
        batch['img_metas'].cpu_only
        Args:
            DataContainer (_type_): _description_
        """

        if isinstance(DataContainer, list):
            out = []
            for dc in DataContainer:
                # print(dc)
                if isinstance(dc.data,list):
                    out.extend(dc.data)
                else:
                    out.append(dc.data)
            return out
        else:
            return DataContainer.data


class BaseDCDataModuleV2(BaseDCDataModule):

    """
    1、 验证集添加 batch
    2、 验证集和测试集的预处理是 训练集逻辑
    """
    def __init__(self, cfg) -> None:
        super().__init__(cfg) 

        train_pipelines = self.hparams.cfg.train['pipeline']
        valid_piplines=[]
        valid_types = ['ImageFromFile','Annotations','Normalize','FormatBundle']
        CollectKeys = dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor',
                           'img_norm_cfg'))
        
        for train_pipeline in train_pipelines:
            for valid_type in valid_types:
                if valid_type in train_pipeline['type']:
                   valid_piplines.append(train_pipeline)

        valid_piplines.append(CollectKeys)
 
        self.hparams.cfg.val['pipeline'] = valid_piplines
        self.hparams.cfg.test['pipeline'] = valid_piplines
        

    def val_dataloader(self) -> DataLoader:
        # The specific dataloader settings
        val_loader_cfg = {
            **self.loader_cfg,
            # 'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **self.hparams.cfg.get('val_dataloader', {}),
        }
        return build_dataloader(self.val_dataset, **val_loader_cfg)

    def test_dataloader(self) -> DataLoader:

        test_loader_cfg = {
            **self.loader_cfg,
            # 'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **self.hparams.cfg.get('test_dataloader', {})
        }
        return build_dataloader(self.test_dataset, **test_loader_cfg)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        out = dict()
        for k, v in batch.items():
            new_v = self.DC2Tensor(v)
            new_v = new_v[0] if k == 'img_metas' else new_v[0]
            out.update({k: new_v})
        return out