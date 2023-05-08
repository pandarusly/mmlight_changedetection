# model settings
mid_channel=128
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DIEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='PDACN',
        backbone=dict(
            type= "mit_b0",
            ),
    seghead=dict(
        type="SimpleFuse",
        in_channels=[32,64,160,256],        
        channels= 128, ),
    changehead=dict(
        type='ChangeSeg',
        change=dict(type="LKADIFF",
                dim=mid_channel,
                k=5,
                td=True,
                k2=3,),          
        fcn=dict(type='fcn',
                in_channels=mid_channel,
                channels=mid_channel))
                
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