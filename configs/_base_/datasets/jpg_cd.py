# dataset settings
dataset_type = 'LEVIR_CD_DatasetJPG'
data_root = 'data/CD_Data_GZ_256'
# img_norm_cfg = dict( mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# +-----------+--------+-----------+--------+-------+-------+
# |   Class   | Fscore | Precision | Recall |  IoU  |  Acc  |
# +-----------+--------+-----------+--------+-------+-------+
# | unchanged | 99.48  |   99.41   | 99.54  | 98.96 | 99.54 |
# |  changed  | 90.09  |   91.29   | 88.93  | 81.98 | 88.93 |
# +-----------+--------+-----------+--------+-------+-------+
img_norm_cfg = dict( mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# +-----------+--------+-----------+--------+-------+-------+
# |   Class   | Fscore | Precision | Recall |  IoU  |  Acc  |
# +-----------+--------+-----------+--------+-------+-------+
# | unchanged | 99.57  |    99.5   | 99.64  | 99.14 | 99.64 |
# |  changed  | 91.85  |   93.09   | 90.63  | 84.92 | 90.63 |
# +-----------+--------+-----------+--------+-------+-------+
crop_size = (256, 256)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    # dict(type='MultiImgShiftOnes',shiftsize=15),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(256, 256),
        # img_scale=(512, 512),
        # img_ratios=[0.75, 1.0, 1.25],
        flip=False,
        transforms=[
            dict(type='MultiImgResize', keep_ratio=True),
            dict(type='MultiImgRandomFlip'),
            dict(type='MultiImgNormalize', **img_norm_cfg),
            dict(type='MultiImgImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='label',
        split='list/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='label',
        split='list/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='label',
        split='list/test.txt',
        pipeline=test_pipeline))