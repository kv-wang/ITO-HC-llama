#!/bin/bash

# 训练identity_tanh_offdiag模型，gamma=0.001
# 在tmux会话中运行

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置参数
export NGPU=2
export METHODS="identity_tanh_offdiag"
export HC_OFFDIAG_INIT_SCALE="0.001"  # gamma初始值
export SAVE_H_RES_FREQ="100"  # 每100步保存H_res矩阵

echo "====================================================================="
echo "Starting training with:"
echo "  Method: identity_tanh_offdiag"
echo "  Gamma init: ${HC_OFFDIAG_INIT_SCALE}"
echo "  GPUs: ${NGPU}"
echo "  H_res save frequency: ${SAVE_H_RES_FREQ} steps"
echo "  Wandb: enabled"
echo "====================================================================="

# 运行训练
./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --metrics.wandb_comment "_gamma${HC_OFFDIAG_INIT_SCALE}" \
    --training.steps 10000 \
    "$@"
