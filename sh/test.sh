# config=logs/train/runs/2023-05-18_10-14-55/MSTACDN_Noneattn_segb0.py
# ckpt=logs/train/runs/2023-05-18_10-14-55/checkpoints/epoch_104_mIoU0.849.ckpt

# python  test.py ${config} ${ckpt}  --work-dir logs/res/  --eval mFscore mIoU
# python  test.py ${config} ${ckpt}  --work-dir logs/res/  --eval-options  imgfile_prefix=logs/res/MSTACDN_Noneattn_segb0 --format-only


# config=logs/train/runs/2023-05-18_10-14-39/MSTACDN_Noneattn_segb0_stm16.py
# ckpt=logs/train/runs/2023-05-18_10-14-39/checkpoints/epoch_158_mIoU0.834.ckpt


config=logs/train/runs/2023-05-17_21-46-27/bit-r18.py
ckpt=logs/train/runs/2023-05-17_21-46-27/checkpoints/last.ckpt #epoch_060_mIoU0.837_in100.ckpt

# python  test.py ${config} ${ckpt}  --work-dir logs/res/  --eval mFscore mIoU
python  test.py ${config} ${ckpt}  --work-dir logs/res/  --eval-options  imgfile_prefix=logs/res/bit_last --format-only