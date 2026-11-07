#!/bin/bash
# 临时启动脚本 - 在tmux中交互式输入WANDB_API_KEY

SESSION_NAME="llama_gamma001"

# 杀掉旧会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing old session..."
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

# 创建新会话，运行交互式设置
tmux new-session -d -s "$SESSION_NAME"

# 发送命令到tmux
tmux send-keys -t "$SESSION_NAME" "cd /home/sfong7/ITO-HC-llama" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Please enter your WANDB_API_KEY:'" C-m
tmux send-keys -t "$SESSION_NAME" "read -s WANDB_API_KEY" C-m

# 等待用户输入后发送训练命令 (用户需要在tmux中按Enter后手动运行下面的命令)
cat << 'EOF' > /tmp/train_cmd.sh
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export NGPU=2
export METHODS="identity_tanh_offdiag"
export HC_OFFDIAG_INIT_SCALE="0.001"
export SAVE_H_RES_FREQ="100"
echo "====="
echo "Training started at: $(date)"
echo "Method: identity_tanh_offdiag, Gamma: 0.001"
echo "====="
./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --metrics.wandb_comment "_gamma0.001" \
    --training.steps 10000 \
    2>&1 | tee train_gamma001.log
echo ""
echo "Training finished at: $(date)"
bash
EOF

echo ""
echo "Tmux session created. Now:"
echo "1. Run: tmux attach -t $SESSION_NAME"
echo "2. Enter your WANDB_API_KEY when prompted (input will be hidden)"
echo "3. Press Enter"
echo "4. Run: bash /tmp/train_cmd.sh"
echo ""
echo "Or use the one-liner:"
echo "  tmux attach -t $SESSION_NAME"
