#!/bin/bash

# 示例：使用identity_tanh_offdiag训练并记录gamma值和H_res矩阵
#
# 功能：
# 1. 使用identity_tanh_offdiag方法进行训练
# 2. 将所有层的gamma值记录到wandb
# 3. 每100步保存一次H_res矩阵到本地
#
# 使用方法：
#   ./example_train_with_h_res_logging.sh

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
NGPU=2  # GPU数量
METHODS="identity_tanh_offdiag"  # 使用identity_tanh_offdiag方法
SAVE_H_RES_FREQ=100  # 每100步保存一次H_res矩阵

# 运行训练
# - 使用AdamW优化器
# - 启用wandb记录
# - gamma初始值为0.05
# - gamma可训练
# - 每100步保存H_res矩阵
NGPU=${NGPU} METHODS=${METHODS} \
    ./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --model.mhc_h_res_offdiag_init_scale 0.05 \
    --model.save_h_res_freq ${SAVE_H_RES_FREQ} \
    --training.steps 1000 \
    "$@"

echo "训练完成！"
echo "Gamma值已记录到wandb"
echo "H_res矩阵已保存到: ./outputs/checkpoints/h_res_matrices/"
