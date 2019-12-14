#!usr/bin/bash

# 训练脚本参数规约:
#
#	$1 跑 CUDA 块号;
#	$2 任务执行日期;

# 激活执行环境.
source ~/py0.3torch0.4/bin/activate
mkdir -p save/$2
touch save/$2/bert_best.pt
touch save/$2/model.pt

# 执行训练程序
CUDA_VISIBLE_DEVICES=$1 python -u train.py \
        -od outputs/result/$2 \
        -sd save/$2/