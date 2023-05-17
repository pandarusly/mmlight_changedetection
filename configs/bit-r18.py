_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/benchmarks/bit_r18.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]
# 
# model=dict(
#     pretrained = 
# )

# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs/bit-r18.py --output-dir browser_jpg_cd
# python get_flops.py configs/bit-r18.py

# python  train.py  configs/bit-r18.py --work-dir logs\fp16_LEVIR_pdacn --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/bit-r18.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py configs/bit-r18.py logs/train/runs/2023-05-17_21-46-27/checkpoints/epoch_127_mIoU0.842.ckpt --eval mFscore mIoU
 
