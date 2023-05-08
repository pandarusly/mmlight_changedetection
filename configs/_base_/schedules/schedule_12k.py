# optimizer
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.001)
    # fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
    # fp16 placeholder
fp16 = dict()
# learning policy
lr_config = dict(policy='poly',power=1,min_lr=1.0e-07,by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=12000)
checkpoint_config = dict(by_epoch=False,max_keep_ckpts=5,save_optimizer=False, interval=1200)
evaluation = dict(interval=1200, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='Fscore.changed', greater_keys=['Fscore'])