_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/benchmarks/pdacn.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

# model=dict(
#     pretrained = 
# )

# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs/MSTACDN.py --output-dir browser_jpg_cd
# python get_flops.py configs/MSTACDN.py

# python  train.py  configs/MSTACDN.py --work-dir logs\MSTACDN --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/MSTACDN.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py logs/train/runs/2023-05-18_01-21-31/MSTACDN.py logs/train/runs/2023-05-18_01-21-31/checkpoints/epoch_238_mIoU0.865.ckpt --work-dir logs/res/
# --eval-options  imgfile_prefix=logs/res/MSTACDN-best --format-only 
# --eval mFscore mIoU