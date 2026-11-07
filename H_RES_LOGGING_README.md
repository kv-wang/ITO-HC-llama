# H_res矩阵和Gamma值记录功能

## 功能说明

针对 `identity_tanh_offdiag` 方法，添加了以下两个功能：

### 1. Wandb记录Gamma值变化

在训练过程中，会自动收集所有层的gamma值并记录到wandb。每个层的attention和FFN都有独立的gamma值。

**记录的指标名称：**
- `gamma/layer_0_attn` - 第0层attention的gamma值
- `gamma/layer_0_ffn` - 第0层FFN的gamma值
- `gamma/layer_1_attn` - 第1层attention的gamma值
- 等等...

### 2. 保存H_res矩阵到本地

每隔指定的步数，将所有层的H_res矩阵保存到本地文件系统。

**保存位置：**
```
<checkpoint_folder>/h_res_matrices/h_res_step_<step>/h_res_matrices.json
```

**保存内容：**
- 每个层的attention H_res矩阵
- 每个层的FFN H_res矩阵
- 格式为JSON，便于后续分析

## 使用方法

### 启用Gamma记录到Wandb

只需确保启用wandb即可自动记录gamma值：

```bash
--metrics.enable_wandb
```

### 启用H_res矩阵保存

添加以下参数来设置保存频率：

```bash
--model.save_h_res_freq 100  # 每100步保存一次H_res矩阵
```

设置为0则禁用保存功能（默认值）。

### 完整示例

修改 `run_llama_2_7b_hc_adamw.sh` 或直接运行：

```bash
METHODS="identity_tanh_offdiag" \
    ./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --model.save_h_res_freq 100 \
    --training.steps 1000
```

## 实现细节

### 1. 配置文件修改

在 `torchtitan/config_manager.py` 中添加了新的配置选项：

```python
self.parser.add_argument(
    "--model.save_h_res_freq",
    type=int,
    default=0,
    help="Save H_res matrices every N steps (0 to disable). Only for identity_tanh_offdiag.",
)
```

### 2. 训练脚本修改

在 `train.py` 中添加了两个新函数：

#### `collect_gamma_values(model, model_config)`
- 遍历所有transformer层
- 从每个层的 `hc_attn` 和 `hc_ffn` 中提取gamma值
- gamma值从log空间转换：`gamma = exp(h_res_offdiag_log_scale)`
- 返回字典格式，键名为 `gamma/layer_{id}_{attn|ffn}`

#### `save_h_res_matrices(model, model_config, step, save_dir)`
- 调用每个HyperConnection的 `h_res_static()` 方法获取H_res矩阵
- H_res矩阵计算公式：`H_res = I + gamma * offdiag * tanh(static_alpha_res)`
- 保存为JSON格式，包含所有层的矩阵数据

### 3. 训练循环集成

在日志记录部分（每 `log_freq` 步）：

```python
# 收集gamma值并添加到wandb日志
if model_config.hyper_conn_type == "identity_tanh_offdiag":
    gamma_values = collect_gamma_values(model_parts[0], model_config)
    wandb_log_dict.update(gamma_values)

wandb.log(wandb_log_dict, step=train_state.step)
```

在适当位置保存H_res矩阵：

```python
# 根据save_h_res_freq保存H_res矩阵
if (h_res_save_dir is not None and 
    job_config.model.save_h_res_freq > 0 and
    train_state.step % job_config.model.save_h_res_freq == 0 and
    global_rank == 0):
    save_h_res_matrices(model_parts[0], model_config, train_state.step, h_res_save_dir)
```

## 查看结果

### Wandb中查看Gamma值

1. 登录wandb网站
2. 进入对应的project和run
3. 在Charts中搜索 "gamma" 即可看到所有层的gamma变化曲线
4. 可以对比不同层的gamma演化趋势

### 读取保存的H_res矩阵

使用Python读取：

```python
import json
import numpy as np

# 读取某一步的H_res矩阵
with open('checkpoints/h_res_matrices/h_res_step_1000/h_res_matrices.json', 'r') as f:
    h_res_data = json.load(f)

# 获取第0层attention的H_res矩阵
h_res_0_attn = np.array(h_res_data['layer_0_attn'])
print(f"Shape: {h_res_0_attn.shape}")
print(f"Diagonal: {np.diag(h_res_0_attn)}")
print(f"Off-diagonal mean: {(h_res_0_attn.sum() - np.trace(h_res_0_attn)) / (h_res_0_attn.size - h_res_0_attn.shape[0])}")
```

### 分析H_res矩阵演化

可以编写脚本比较不同步数的H_res矩阵，分析：
- 对角线值的变化（应该接近1）
- 非对角线值的演化（受gamma和tanh(logits)影响）
- 不同层之间的差异
- 训练过程中矩阵的稳定性

## 性能考虑

- **Gamma记录**: 开销极小，只是读取参数值并记录，不影响训练速度
- **H_res矩阵保存**: 
  - 只在指定步数执行
  - 只在rank 0上执行
  - 矩阵较小（n×n，默认n=4），IO开销可忽略
  - 建议保存频率不要太高（如每100-1000步）

## 注意事项

1. **仅支持identity_tanh_offdiag方法**: 其他hyper-connection方法不会记录gamma或保存H_res
2. **分布式训练**: H_res保存只在rank 0执行，避免重复保存
3. **磁盘空间**: 注意保存频率和训练步数，避免产生过多文件
4. **checkpoint兼容性**: H_res保存不影响checkpoint的加载和恢复

## 故障排查

如果gamma值没有出现在wandb中：
1. 确认使用的是 `identity_tanh_offdiag` 方法
2. 确认 `--metrics.enable_wandb` 已启用
3. 检查是否在rank 0上运行（多GPU时）

如果H_res矩阵没有保存：
1. 确认 `--model.save_h_res_freq` 大于0
2. 确认训练步数已达到保存频率的倍数
3. 检查checkpoint目录的写权限
