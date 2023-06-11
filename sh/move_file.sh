#!/bin/bash

# 创建test目录
mkdir -p mscdn

# 读取txt文件的内容，并逐行处理
while IFS= read -r line; do
    # 处理A和B文件夹中的jpg文件
    if [[ -f "/root/mmlight_changedetection/data/CD_Data_GZ_256/A/$line.jpg" ]]; then
        cp "/root/mmlight_changedetection/data/CD_Data_GZ_256/A/$line.jpg" "mscdn/${line}_A.jpg"
    fi

    if [[ -f "/root/mmlight_changedetection/data/CD_Data_GZ_256/B/$line.jpg" ]]; then
        cp "/root/mmlight_changedetection/data/CD_Data_GZ_256/B/$line.jpg" "mscdn/${line}_B.jpg"
    fi

    # 处理label文件夹中的png文件
    if [[ -f "/root/mmlight_changedetection/data/CD_Data_GZ_256/label/$line.png" ]]; then
        cp "/root/mmlight_changedetection/data/CD_Data_GZ_256/label/$line.png" "mscdn/${line}_L.png"
    fi

    # 处理实验中的结果
    if [[ -f "/root/mmlight_changedetection/logs/res/bit-r18/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/bit-r18/$line.png" "mscdn/${line}_Z_bit.png"
    fi 

    if [[ -f "/root/mmlight_changedetection/logs/res/changeformer_mit-b0/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/changeformer_mit-b0/$line.png" "mscdn/${line}_Z_changeformer.png"
    fi

    if [[ -f "/root/mmlight_changedetection/logs/res/fc_siam_diff/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/fc_siam_diff/$line.png" "mscdn/${line}_Z_FCSD.png"
    fi

    if [[ -f "/root/mmlight_changedetection/logs/res/snunet_c16/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/snunet_c16/$line.png" "mscdn/${line}_Z_SUNet.png"
    fi

    #  mscdn实验
     if [[ -f "/root/mmlight_changedetection/logs/res/MSTACDN_attn_segb0/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/MSTACDN_attn_segb0/$line.png" "mscdn/${line}_Z_MSTACDN_attn_segb0.png"
    fi

    if [[ -f "/root/mmlight_changedetection/logs/res/MSTACDN_attn_segb0_stm/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/MSTACDN_attn_segb0_stm/$line.png" "mscdn/${line}_Z_MSTACDN_attn_segb0_stm.png"
    fi

    if [[ -f "/root/mmlight_changedetection/logs/res/MSTACDN_Noneattn_segb0_256/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/MSTACDN_Noneattn_segb0_256/$line.png" "mscdn/${line}_Z_MSTACDN_Noneattn_segb0_256.png"
    fi

    if [[ -f "/root/mmlight_changedetection/logs/res/MSTACDN-best/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/MSTACDN-best/$line.png" "mscdn/${line}_Z_MSTACDN-best.png"
    fi

    if [[ -f "/root/mmlight_changedetection/logs/res/MSTACDN_Noneattn_segb0/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/MSTACDN_Noneattn_segb0/$line.png" "mscdn/${line}_Z_MSTACDN_Noneattn_segb0.png"
    fi

    if [[ -f "/root/mmlight_changedetection/logs/res/MSTACDN_Noneattn_segb0_stm16/$line.png" ]]; then
        cp "/root/mmlight_changedetection/logs/res/MSTACDN_Noneattn_segb0_stm16/$line.png" "mscdn/${line}_Z_MSTACDN_Noneattn_segb0_stm16.png"
    fi

done < /root/mmlight_changedetection/data/CD_Data_GZ_256/list/test.txt