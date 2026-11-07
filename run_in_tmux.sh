#!/bin/bash
# 这个脚本在tmux内部运行
# 使用方法: 先 export WANDB_API_KEY="your_key"，然后运行此脚本

cd /home/sfong7/ITO-HC-llama

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY not set"
    echo "Please run: export WANDB_API_KEY='your_key_here'"
    echo "Then run this script again"
    exit 1
fi

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export NGPU=2
export METHODS="identity_tanh_offdiag"
export HC_OFFDIAG_INIT_SCALE="0.001"
export SAVE_H_RES_FREQ="100"

echo "====================================================================="
echo "Starting Identity Tanh Offdiag Training"
echo "====================================================================="
echo "Time: $(date)"
echo "Method: $METHODS"
echo "Gamma: $HC_OFFDIAG_INIT_SCALE"
echo "GPUs: $NGPU"
echo "H_res save frequency: $SAVE_H_RES_FREQ steps"
echo "Wandb: enabled (base: $WANDB_BASE_URL)"
echo "====================================================================="
echo ""

./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --metrics.wandb_comment "_gamma0.001" \
    --training.steps 10000 \
    2>&1 | tee train_gamma001.log

echo ""
echo "====================================================================="
echo "Training finished at: $(date)"
echo "====================================================================="
