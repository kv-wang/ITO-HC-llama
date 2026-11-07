#!/bin/bash

# 交互式启动训练脚本
# 会提示输入WANDB_API_KEY（如果未设置）

set -e

PROJECT_DIR="/home/sfong7/ITO-HC-llama"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "Identity Tanh Offdiag Training Launcher (gamma=0.001)"
echo "======================================================================"
echo ""

# 检查并获取WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY is not set in environment."
    echo ""
    read -p "Enter your WANDB_API_KEY (or press Ctrl+C to cancel): " api_key

    if [ -z "$api_key" ]; then
        echo "Error: API key cannot be empty"
        exit 1
    fi

    export WANDB_API_KEY="$api_key"
    echo ""
    echo "✓ WANDB_API_KEY set for this session"
else
    echo "✓ WANDB_API_KEY found in environment"
fi

echo ""
echo "Training configuration:"
echo "  Method: identity_tanh_offdiag"
echo "  Gamma init: 0.001"
echo "  GPUs: 2"
echo "  Optimizer: AdamW"
echo "  Training steps: 10000"
echo "  H_res save freq: 100 steps"
echo "  Wandb: enabled"
echo ""

# 询问是否继续
read -p "Start training in tmux? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

SESSION_NAME="llama_gamma001"

# 检查是否已有同名会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo ""
    echo "Warning: Tmux session '$SESSION_NAME' already exists."
    read -p "Kill and restart? (y/n): " restart

    if [ "$restart" = "y" ] || [ "$restart" = "Y" ]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "Old session killed."
    else
        echo "Attaching to existing session..."
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi
fi

echo ""
echo "Starting training in tmux session '$SESSION_NAME'..."

# 创建训练命令
TRAIN_CMD="cd $PROJECT_DIR && \
export WANDB_API_KEY='$WANDB_API_KEY' && \
export WANDB_BASE_URL='${WANDB_BASE_URL:-https://api.bandw.top}' && \
export NGPU=2 && \
export METHODS='identity_tanh_offdiag' && \
export HC_OFFDIAG_INIT_SCALE='0.001' && \
export SAVE_H_RES_FREQ='100' && \
echo '=====================================================================' && \
echo 'Training started at: '\$(date) && \
echo '=====================================================================' && \
./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --metrics.wandb_comment '_gamma0.001' \
    --training.steps 10000 \
    2>&1 | tee train_gamma001.log; \
echo ''; \
echo 'Training finished at: '\$(date); \
echo 'Press Enter to close this tmux session...'; \
read"

# 启动tmux会话
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_CMD"

# 等待确认启动
sleep 2

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo ""
    echo "======================================================================"
    echo "✓ Training started successfully!"
    echo "======================================================================"
    echo ""
    echo "View progress:"
    echo "  tmux attach -t $SESSION_NAME"
    echo ""
    echo "View log:"
    echo "  tail -f train_gamma001.log"
    echo ""
    echo "Stop training:"
    echo "  tmux kill-session -t $SESSION_NAME"
    echo ""
    echo "Detach from tmux: Ctrl+B, then D"
    echo "======================================================================"
    echo ""

    read -p "Attach to tmux session now? (y/n): " attach
    if [ "$attach" = "y" ] || [ "$attach" = "Y" ]; then
        tmux attach -t "$SESSION_NAME"
    fi
else
    echo ""
    echo "✗ Training session exited immediately. Checking log..."
    echo ""
    tail -30 train_gamma001.log
    exit 1
fi
