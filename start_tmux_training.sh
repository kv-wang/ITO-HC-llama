#!/bin/bash

# 在tmux中启动identity_tanh_offdiag训练 (gamma=0.001)
#
# 使用方法:
#   1. 设置WANDB_API_KEY环境变量
#      export WANDB_API_KEY="your_key_here"
#
#   2. 运行此脚本
#      ./start_tmux_training.sh
#
#   3. 查看训练进度
#      tmux attach -t llama_gamma001
#      或查看日志: tail -f train_gamma001.log

set -e

# 检查WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set"
    echo ""
    echo "Please set it first:"
    echo "  export WANDB_API_KEY=\"your_api_key\""
    echo ""
    echo "Then run this script again."
    exit 1
fi

# 项目目录
PROJECT_DIR="/home/sfong7/ITO-HC-llama"
cd "$PROJECT_DIR"

# Tmux会话名
SESSION_NAME="llama_gamma001"

# 检查会话是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "Do you want to:"
    echo "  1) Attach to existing session (tmux attach -t $SESSION_NAME)"
    echo "  2) Kill and restart"
    read -p "Enter choice (1/2): " choice

    if [ "$choice" = "2" ]; then
        echo "Killing existing session..."
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Attaching to existing session..."
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi
fi

# 创建训练启动命令
TRAIN_CMD="cd $PROJECT_DIR && \
export WANDB_API_KEY='$WANDB_API_KEY' && \
export WANDB_BASE_URL='${WANDB_BASE_URL:-https://api.bandw.top}' && \
export NGPU=2 && \
export METHODS='identity_tanh_offdiag' && \
export HC_OFFDIAG_INIT_SCALE='0.001' && \
export SAVE_H_RES_FREQ='100' && \
echo '=====================================================================' && \
echo 'Starting training with:' && \
echo '  Method: identity_tanh_offdiag' && \
echo '  Gamma init: 0.001' && \
echo '  GPUs: 2' && \
echo '  H_res save frequency: 100 steps' && \
echo '  Wandb: enabled' && \
echo '=====================================================================' && \
./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --metrics.wandb_comment '_gamma0.001' \
    --training.steps 10000 \
    2>&1 | tee train_gamma001.log"

# 创建tmux会话
echo "Creating tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_CMD"

# 等待一下让训练开始
sleep 3

# 检查会话是否还在运行
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo ""
    echo "✓ Training started successfully in tmux session '$SESSION_NAME'"
    echo ""
    echo "To view training progress:"
    echo "  tmux attach -t $SESSION_NAME"
    echo ""
    echo "To detach from tmux session: Ctrl+B, then D"
    echo ""
    echo "To view log file:"
    echo "  tail -f $PROJECT_DIR/train_gamma001.log"
    echo ""
    echo "To stop training:"
    echo "  tmux kill-session -t $SESSION_NAME"
    echo ""
else
    echo ""
    echo "✗ Training session exited. Check log for errors:"
    echo "  cat $PROJECT_DIR/train_gamma001.log"
    echo ""
    exit 1
fi
