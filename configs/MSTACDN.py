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
# python get_flops.py configs/changeformer_mit-b0.py

# python  train.py  configs/MSTACDN.py --work-dir logs\MSTACDN --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/MSTACDN.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py configs/MSTACDN.py F:\study-note\python-note\Advance_py\DLDEV\Task-36869338\代码截图\代码截图\论文代码\log\LeVirCd\MDACN\2022-07-25_11-45-23\checkpoints\opencd_mdac_f10.9227_levir.pth --eval mFscore mIoU
 
