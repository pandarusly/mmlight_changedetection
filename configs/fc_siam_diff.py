_base_ = [ 
    './_base_/datasets/jpg_cd.py',
    './_base_/model/benchmarks/fc_siam_diff.py',
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

# python browse_dataset.py configs/fc_siam_diff.py --output-dir browser_jpg_cd
# python get_flops.py configs/fc_siam_diff.py

# python  train.py  configs/fc_siam_diff.py --work-dir logs\fp16_LEVIR_pdacn --gpu-id 0 --seed 42 
# python  train_lightning.py cfg_path=configs/fc_siam_diff.py  experiment=epoch_300_adamw_poly  logger=wandb

# python  test.py configs/fc_siam_diff.py F:\study-note\python-note\Advance_py\DLDEV\Task-36869338\代码截图\代码截图\论文代码\log\LeVirCd\MDACN\2022-07-25_11-45-23\checkpoints\opencd_mdac_f10.9227_levir.pth --eval mFscore mIoU
 
