model = dict(
    type='SiamEncoderDecoder',
    pretrained='/root/mmlight_changedetection/data/mit_b0.pth',
    backbone=dict(
        type= "mit_b0"
        ),
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        type='TIAHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        fuse_cfg=dict(
            type='FPNFuse',
            in_channels=[32, 64, 160, 256],
            mid_channel_3x3=128,
            out_channel_3x3=128,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        attn=dict(type='LAKForChange',dim=128),
        difference_cfg=dict(type='abs_diff', i_c=128),
        channels=128,
        out_channels=2,
        num_classes=2,
        threshold=0.5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))