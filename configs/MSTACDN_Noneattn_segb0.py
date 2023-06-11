_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/base_attn_segb0.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]


model = dict(
     
    decode_head=dict(
        in_channels=[32, 64, 160, 256],
        attn=None,
        difference_cfg=dict(type='abs_diff', i_c=128),
        channels=128,
        out_channels=2,
        num_classes=2,
        ),
                        )

    
# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs/MSTACDN_Noneattn_segb0.py --output-dir browser_jpg_cd
# python get_flops.py configs/MSTACDN_Noneattn_segb0.py

# python  train.py  configs/MSTACDN_Noneattn_segb0.py --work-dir logs\MSTACDN_Noneattn_segb0 --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/MSTACDN_Noneattn_segb0.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py logs/train/runs/2023-05-18_06-15-16/MSTACDN_Noneattn_segb0.py logs/train/runs/2023-05-18_06-15-16/checkpoints/epoch_185_mIoU0.847.ckpt  --work-dir logs/res/
# --eval-options  imgfile_prefix=logs/res/MSTACDN_Noneattn_segb0 --format-only 
# --eval mFscore mIoU