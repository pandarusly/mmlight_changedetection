_base_ = [ 
    './_base_/datasets/linux_levir_256.py',
    './_base_/model/benchmarks/pdacn.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]


# img_norm_cfg = dict( mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# +-----------+--------+-----------+--------+-------+-------+
# |   Class   | Fscore | Precision | Recall |  IoU  |  Acc  |
# +-----------+--------+-----------+--------+-------+-------+
# | unchanged | 99.48  |   99.41   | 99.54  | 98.96 | 99.54 |
# |  changed  | 90.09  |   91.29   | 88.93  | 81.98 | 88.93 |
# +-----------+--------+-----------+--------+-------+-------+
# img_norm_cfg = dict( mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# +-----------+--------+-----------+--------+-------+-------+
# |   Class   | Fscore | Precision | Recall |  IoU  |  Acc  |
# +-----------+--------+-----------+--------+-------+-------+
# | unchanged | 99.57  |    99.5   | 99.64  | 99.14 | 99.64 |
# |  changed  | 91.85  |   93.09   | 90.63  | 84.92 | 90.63 |
# +-----------+--------+-----------+--------+-------+-------+



# model=dict(
#     pretrained = 
# )

# custom_hooks = [
#     dict(type='CustomEvalHookV2',iterval=120,num_eval_images=4,priority='VERY_LOW')
# ]

# python browse_dataset.py configs\fp16_LEVIR_pdacn.py --output-dir   browser_linux_levir_256

# python  train.py  configs\fp16_LEVIR_pdacn.py --work-dir logs\fp16_LEVIR_pdacn --gpu-id 0 --seed 42

# python  test.py configs\fp16_LEVIR_pdacn.py F:\study-note\python-note\Advance_py\DLDEV\Task-36869338\代码截图\代码截图\论文代码\log\LeVirCd\MDACN\2022-07-25_11-45-23\checkpoints\opencd_mdac_f10.9227_levir.pth --eval mFscore mIoU
 
