_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/base_attn.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

model = dict(  
    decode_head=dict(attn=None),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

    
# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs/MSTACDN_Noattn.py --output-dir browser_jpg_cd
# python get_flops.py configs/MSTACDN_Noattn.py

# python  train.py  configs/MSTACDN_Noattn.py --work-dir logs\MSTACDN_Noattn --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/MSTACDN_Noattn.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py configs/MSTACDN_Noattn.py logs/train/runs/2023-05-17_22-09-10/checkpoints/epoch_244_mIoU0.779.ckpt   --work-dir logs/res/
# --eval-options  imgfile_prefix=logs/res/MSTACDN_Noattn --format-only 
# --eval mFscore mIoU