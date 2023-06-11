_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/base_attn_segb0.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

# model = dict()

    
# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs/MSTACDN_attn_segb0.py --output-dir browser_jpg_cd
# python get_flops.py configs/MSTACDN_attn_segb0.py

# python  train.py  configs/MSTACDN_attn_segb0.py --work-dir logs\MSTACDN_attn_segb0 --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/MSTACDN_attn_segb0.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py logs/train/runs/2023-05-18_08-33-59/MSTACDN_attn_segb0.py logs/train/runs/2023-05-18_08-33-59/checkpoints/epoch_067_mIoU0.844.ckpt  --work-dir logs/res/
# --eval-options  imgfile_prefix=logs/res/MSTACDN_attn_segb0 --format-only 
# --eval mFscore mIoU



# python trace.py logs/train/runs/2023-05-18_08-33-59/MSTACDN_attn_segb0.py --checkpoint logs/train/runs/2023-05-18_08-33-59/checkpoints/epoch_067_mIoU0.844.ckpt --shape 256 256 --output-file logs/google.pth