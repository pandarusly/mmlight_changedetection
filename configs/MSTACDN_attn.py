_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/base_attn.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

# model = dict()

    
# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs/MSTACDN_attn.py --output-dir browser_jpg_cd
# python get_flops.py configs/MSTACDN_attn.py

# python  train.py  configs/MSTACDN_attn.py --work-dir logs\MSTACDN_attn --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/MSTACDN_attn.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py configs/MSTACDN_attn.py logs/train/runs/2023-05-18_01-32-45/checkpoints/epoch_032_mIoU0.768.ckpt   --work-dir logs/res/
# --eval-options  imgfile_prefix=logs/res/MSTACDN_attn --format-only 
# --eval mFscore mIoU