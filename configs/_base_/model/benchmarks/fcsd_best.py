# model settings
mid_channel=128
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DIEncoderDecoder',
    pretrained='data/pdacn_delmodel.pth',
    backbone=dict(
        type='FCSiamDiff', 
        num_band=3, 
        num_class=2,
    ),
    decode_head=dict(
        type='IdentityHead',
        in_channels=1,
        in_index=-1,
        num_classes=2,
        out_channels=2, # support single class
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))