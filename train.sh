#!/bin/bash

# 1. 获取参数，并设置默认值（可选）
# 如果运行时不传参数，这里会报错退出并提示
data=${1:?"请提供 data 参数 (e.g. room)"}
cuda=${2:-0}  # 如果不传 cuda 参数，默认使用 0 号显卡

# 2. 执行深度图处理
# 建议将路径写成变量，方便维护
BASE_PATH="/home/wxy/3DGS/CL-GS/data/PINHOLE/${data}"

echo "Starting depth scale processing for: ${data}..."

python utils/make_depth_scale.py \
    --base_dir "${BASE_PATH}" \
    --depths_dir "${BASE_PATH}/depth" \
    --model_type bin && \
\
# 3. 执行训练
echo "Starting training on CUDA device: ${cuda}..."

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    -s "${BASE_PATH}" \
    -d "${BASE_PATH}/depth" \
    --exposure_lr_init 0.001 \
    --exposure_lr_final 0.0001 \
    --exposure_lr_delay_steps 5000 \
    --exposure_lr_delay_mult 0.001 \
    --train_test_exp