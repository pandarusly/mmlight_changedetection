# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import build_dataset,DATASETS
 

@DATASETS.register_module()
class SemiDataset:
    def __init__(self, sup_dataset, unsup_dataset, default_args=None,**kwargs):
        self.sup_dataset = build_dataset(sup_dataset, default_args)
        tmp_sup_dataset = self.sup_dataset.datasets[0] if self.sup_dataset.__class__.__name__ == "ConcatDataset" else self.sup_dataset

        self.ignore_index = tmp_sup_dataset.ignore_index
        self.CLASSES = tmp_sup_dataset.CLASSES
        self.PALETTE = tmp_sup_dataset.PALETTE
        
        if unsup_dataset:
            self.unsup_dataset = build_dataset(unsup_dataset, default_args)
            tmp_unsup_dataset = self.unsup_dataset.datasets[0] if self.unsup_dataset.__class__.__name__ == "ConcatDataset" else self.unsup_dataset
            assert tmp_sup_dataset.ignore_index == tmp_unsup_dataset.ignore_index
            assert tmp_sup_dataset.CLASSES == tmp_unsup_dataset.CLASSES
            assert tmp_sup_dataset.PALETTE == tmp_unsup_dataset.PALETTE
        else:
            self.unsup_dataset = None

    # --- ssl by mmseg  jianlong-yuan/semi-mmseg
    # def __len__(self):
    #     return len(self.sup_dataset) * len(self.unsup_dataset)
    # def __getitem__(self, idx):
    #     sup_data = self.sup_dataset[idx // len(self.unsup_dataset)]
    #     unsup_data = self.unsup_dataset[idx % len(self.unsup_dataset)]
    #     return {"sup_data":sup_data, "unsup_data": unsup_data}
    # --- ssl by changedetection 
    def __len__(self):
        if self.unsup_dataset:
            return max(len(self.sup_dataset),len(self.unsup_dataset))
        else:
            return len(self.sup_dataset)
        
    def __getitem__(self, idx):

        sup_data = self.sup_dataset[idx ]  if idx < len(self.sup_dataset) else  self.sup_dataset[idx % len(self.sup_dataset)]
        if self.unsup_dataset:
            unsup_data = self.unsup_dataset[idx] if idx < len(self.unsup_dataset) else  self.unsup_dataset[idx % len(self.unsup_dataset)]
            return {"sup_data":sup_data, "unsup_data": unsup_data}
        else:
            return {"sup_data":sup_data}
 

# dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))

if __name__ == "__main__":
    # from opencd import __version__
    train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='MultiImgNormalize',
        mean=[127.5,127.5,127.5],
        std=[127.5,127.5,127.5],
        to_rgb=True),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ]
    train_sup=dict(
        type='LEVIR_CD_Dataset',
        data_root='data/samples_LEVIR',
        img_dir='',
        ann_dir='label',
        split='list/train.txt',
        pipeline=train_pipeline) 

    train_unsup_pipeline =  [
    dict(type='MultiImgLoadImageFromFile'), 
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='MultiImgNormalize',
        mean=[127.5,127.5,127.5],
        std=[127.5,127.5,127.5],
        to_rgb=True),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
    ]
    train_unsup=dict(
        type='LEVIR_CD_Dataset',
        data_root='data/samples_LEVIR',
        img_dir='',
        ann_dir='label',
        split='list/test.txt',
        pipeline=train_unsup_pipeline)  

    train=dict(
        type="SemiDataset",
        sup_dataset=train_sup,
        unsup_dataset=train_unsup
    )

    dataset = SemiDataset(sup_dataset=train_sup,
        unsup_dataset=train_unsup)

    print(len(dataset))

    for index in [0,1,2,3]:
        sup_data = dataset[index]['sup_data']
        unsup_data = dataset[index]['unsup_data']

        # dict_keys(['img_metas', 'img']) dict_keys(['img_metas', 'img', 'gt_semantic_seg']
        print(unsup_data.keys(),sup_data.keys())
        print("sup_data img_metas: {} ".format(sup_data['img_metas'].data['filename']))
        print("unsup_data img_metas: {} ".format(unsup_data['img_metas'].data['filename']))

