# 启动训练指南

## 快速启动

### 1. 设置WANDB_API_KEY

```bash
# 在当前shell中设置（仅本次有效）
export WANDB_API_KEY="your_api_key_here"

# 或者添加到 ~/.bashrc 使其永久生效
echo 'export WANDB_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 启动训练

```bash
cd /home/sfong7/ITO-HC-llama
./start_tmux_training.sh
```

### 3. 查看训练进度

```bash
# 方法1: 连接到tmux会话
tmux attach -t llama_gamma001

# 退出tmux（不停止训练）: Ctrl+B, 然后按 D

# 方法2: 查看日志文件
tail -f train_gamma001.log

# 方法3: 查看最近的输出
tmux capture-pane -t llama_gamma001 -p | tail -50
```

### 4. 停止训练

```bash
tmux kill-session -t llama_gamma001
```

## 训练配置

当前配置：
- **方法**: identity_tanh_offdiag
- **Gamma初始值**: 0.001
- **GPU数量**: 2
- **优化器**: AdamW
- **训练步数**: 10000
- **H_res保存频率**: 每100步
- **Wandb**: 已启用

## 输出位置

### Wandb指标
登录到 wandb 查看：
- `gamma/layer_0_attn`, `gamma/layer_0_ffn`, ...
- `loss_metrics/avg_train_loss`
- 其他训练指标

### H_res矩阵文件
```
outputs/checkpoint/<model_config>/h_res_matrices/
  h_res_step_100/h_res_matrices.json
  h_res_step_200/h_res_matrices.json
  ...
```

### 日志文件
```
/home/sfong7/ITO-HC-llama/train_gamma001.log
```

## 自定义训练参数

如果需要修改参数，编辑 `train_identity_tanh_gamma0.001.sh`：

```bash
# 修改GPU数量
export NGPU=4

# 修改gamma初始值
export HC_OFFDIAG_INIT_SCALE="0.01"

# 修改H_res保存频率
export SAVE_H_RES_FREQ="50"

# 修改训练步数
# 在脚本最后的 --training.steps 参数中修改
```

## 故障排除

### 问题1: WANDB_API_KEY未设置
**错误信息**: `WANDB_API_KEY: please set WANDB_API_KEY`

**解决方法**:
```bash
export WANDB_API_KEY="your_key"
./start_tmux_training.sh
```

### 问题2: Tmux会话立即退出
**检查日志**:
```bash
cat train_gamma001.log
```

常见原因：
- WANDB_API_KEY未设置
- 数据路径不正确
- GPU不可用

### 问题3: 无法连接到tmux会话
**检查会话是否存在**:
```bash
tmux ls
```

**如果不存在，检查日志找出原因**:
```bash
cat train_gamma001.log
```

## 监控训练

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看训练loss
```bash
# 实时查看日志中的loss
tail -f train_gamma001.log | grep "loss_metrics"
```

### 查看gamma值变化
登录wandb查看图表，或使用wandb API：
```python
import wandb
api = wandb.Api()
run = api.run("your-entity/your-project/run-id")
history = run.history()
print(history[['step', 'gamma/layer_0_attn']])
```

## 分析H_res矩阵

训练过程中保存的H_res矩阵可以用提供的分析脚本查看：

```bash
# 分析所有保存的H_res矩阵
python analyze_h_res_matrices.py outputs/checkpoint/<config>/h_res_matrices/

# 或手动加载
python3 << EOF
import json
import numpy as np

with open('outputs/checkpoint/.../h_res_step_100/h_res_matrices.json') as f:
    h_res = json.load(f)
    
print("Layers:", list(h_res.keys()))
H = np.array(h_res['layer_0_attn'])
print("Shape:", H.shape)
print("Diagonal:", np.diag(H))
print("Off-diagonal mean:", (H.sum() - np.trace(H)) / (H.size - H.shape[0]))
EOF
```
