# 在Tmux中启动训练 - 快速指南

## 方法1: 最简单 - 直接在现有tmux中运行

```bash
# 1. 连接到tmux会话
tmux attach -t llama_gamma001

# 2. 设置WANDB_API_KEY
export WANDB_API_KEY="your_api_key_here"

# 3. 运行训练
./run_in_tmux.sh

# 4. 训练开始后，按 Ctrl+B 然后 D 来detach
```

## 方法2: 如果WANDB_API_KEY已经在环境中

如果你已经在当前shell中设置了 `WANDB_API_KEY`：

```bash
# 显示给我看你的API key (前10个字符)
echo ${WANDB_API_KEY:0:10}

# 如果显示了内容，我就可以帮你自动启动
```

## 方法3: 永久设置API key

```bash
# 添加到 ~/.bashrc (只需要做一次)
echo 'export WANDB_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc

# 然后可以直接运行
tmux new-session -d -s llama_gamma001 'cd /home/sfong7/ITO-HC-llama && ./run_in_tmux.sh'
tmux attach -t llama_gamma001
```

## 检查训练状态

```bash
# 查看日志
tail -f /home/sfong7/ITO-HC-llama/train_gamma001.log

# 查看tmux会话
tmux ls

# 连接到训练会话
tmux attach -t llama_gamma001

# 查看GPU使用
watch -n 1 nvidia-smi
```

## 当前可用的tmux会话

运行 `tmux ls` 看到：
- llama7b
- llama_gamma001 (准备好了，等待启动训练)
- llama_record

## 训练参数

- Method: identity_tanh_offdiag
- Gamma: 0.001
- GPUs: 2
- Steps: 10000
- H_res保存频率: 100步
- 优化器: AdamW
- Wandb: 启用

## 输出位置

- 日志: `train_gamma001.log`
- H_res矩阵: `outputs/checkpoint/*/h_res_matrices/h_res_step_*/h_res_matrices.json`
- Checkpoints: `outputs/checkpoint/`

---

**下一步**: 请告诉我你的WANDB_API_KEY，或者你可以直接在tmux中设置并运行 `./run_in_tmux.sh`
