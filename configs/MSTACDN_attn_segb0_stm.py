_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/base_attn_segb0.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

model = dict(
     
    decode_head=dict(
        in_channels=[32, 64, 160, 256],
        attn=dict(type='LAKForChange',dim=128),
        difference_cfg=dict(type='abs_diff', i_c=128),
        channels=128,
        out_channels=2,
        num_classes=2,
        ),
        auxiliary_head=dict(type='SimClrHeadForPublic', 
        
                                weight=0.2,
                                loss_decode=dict(
                                type="ContrastiveLossV2",
                                batch_size=16,
                                ),
                                in_channels=256)
                        )
    
# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs/MSTACDN_attn_segb0_stm.py --output-dir browser_jpg_cd
# python get_flops.py configs/MSTACDN_attn_segb0_stm.py

# python  train.py  configs/MSTACDN_attn_segb0_stm.py --work-dir logs\MSTACDN_attn_segb0_stm --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/MSTACDN_attn_segb0_stm.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py logs/train/runs/2023-05-18_08-29-50/MSTACDN_attn_segb0_stm.py logs/train/runs/2023-05-18_08-29-50/checkpoints/epoch_201_mIoU0.845.ckpt  --work-dir logs/res/
# --eval-options  imgfile_prefix=logs/res/MSTACDN_attn_segb0_stm --format-only 
# --eval mFscore mIoU




# python trace.py logs/train/runs/2023-05-18_08-29-50/MSTACDN_attn_segb0_stm.py --checkpoint logs/train/runs/2023-05-18_08-29-50/checkpoints/epoch_201_mIoU0.845.ckpt --shape 256 256 --output-file logs/google_stm.pth