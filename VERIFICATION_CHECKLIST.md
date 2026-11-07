# 功能验证清单

## 实现功能概述

✅ **功能1**: 在wandb上记录gamma的变化
✅ **功能2**: 每隔若干steps保存所有layer的H_res矩阵到本地

## 修改的核心代码

### 1. 配置参数 (torchtitan/config_manager.py)
```python
--model.save_h_res_freq <N>  # 每N步保存H_res（0禁用）
```

### 2. 新增函数 (train.py)

**collect_gamma_values()** - 收集gamma值
- 遍历model.layers
- 从layer.hc_attn.h_res_offdiag_log_scale提取
- 从layer.hc_ffn.h_res_offdiag_log_scale提取
- gamma = exp(h_res_offdiag_log_scale)

**save_h_res_matrices()** - 保存H_res矩阵
- 调用layer.hc_attn.h_res_static()
- 调用layer.hc_ffn.h_res_static()
- 保存为JSON格式

### 3. 集成点

```python
# 初始化（约380行）
if identity_tanh_offdiag and save_h_res_freq > 0:
    创建h_res_save_dir

# 日志记录（约740-765行）
if enable_wandb and rank==0:
    gamma_values = collect_gamma_values()
    wandb.log({...gamma_values...})

# 保存矩阵（约760-765行）
if step % save_h_res_freq == 0 and rank==0:
    save_h_res_matrices()
```

## 使用示例

```bash
# 完整命令
METHODS="identity_tanh_offdiag" \
    ./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --model.save_h_res_freq 100 \
    --training.steps 1000
```

## 预期输出

### Wandb指标
- `gamma/layer_0_attn`: float
- `gamma/layer_0_ffn`: float
- `gamma/layer_1_attn`: float
- ... (每层2个指标)

### 本地文件
```
<checkpoint_folder>/h_res_matrices/
  h_res_step_100/h_res_matrices.json
  h_res_step_200/h_res_matrices.json
  ...
```

### JSON结构
```json
{
  "layer_0_attn": [[1.0, ...], ...],  // shape: (n, n)
  "layer_0_ffn": [[1.0, ...], ...],
  "layer_1_attn": [[1.0, ...], ...],
  ...
}
```

## 验证步骤

### 1. 代码正确性
```bash
# 检查语法
python3 -m py_compile train.py
python3 -m py_compile torchtitan/config_manager.py

# 检查函数存在
grep "def collect_gamma_values" train.py
grep "def save_h_res_matrices" train.py
grep "save_h_res_freq" torchtitan/config_manager.py
```

### 2. 运行时验证
启动一个短训练任务：
```bash
METHODS="identity_tanh_offdiag" \
    ./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --model.save_h_res_freq 10 \
    --training.steps 50
```

检查：
- [ ] 训练正常启动
- [ ] wandb显示gamma指标
- [ ] checkpoint目录下有h_res_matrices文件夹
- [ ] 步数10, 20, 30, 40, 50有对应的JSON文件

### 3. 结果验证
```bash
# 分析保存的矩阵
python analyze_h_res_matrices.py <checkpoint_folder>/h_res_matrices/

# 检查矩阵维度和值
python3 -c "
import json
with open('<path>/h_res_step_10/h_res_matrices.json') as f:
    data = json.load(f)
    print('Keys:', list(data.keys()))
    import numpy as np
    h = np.array(data['layer_0_attn'])
    print('Shape:', h.shape)
    print('Diagonal:', np.diag(h))
"
```

## 技术细节

### H_res计算公式（identity_tanh_offdiag）
```python
H_res = I + gamma * (1 - I) * tanh(static_alpha_res)
```
其中：
- I: 单位矩阵
- gamma: exp(h_res_offdiag_log_scale)
- (1 - I): 非对角线mask
- tanh(static_alpha_res): 静态logits的tanh

### Gamma存储形式
- 参数：`h_res_offdiag_log_scale` (log空间)
- 实际值：`gamma = exp(h_res_offdiag_log_scale)`
- 初始值：log(0.05) ≈ -3.0

## 辅助工具

1. **analyze_h_res_matrices.py** - 分析和可视化H_res矩阵
2. **example_train_with_h_res_logging.sh** - 完整的训练示例
3. **H_RES_LOGGING_README.md** - 详细文档

## 常见问题

**Q: wandb上看不到gamma指标？**
A: 检查：
1. 使用的是identity_tanh_offdiag方法
2. --metrics.enable_wandb已启用
3. 当前是rank 0进程

**Q: H_res没有保存？**
A: 检查：
1. --model.save_h_res_freq > 0
2. 训练步数达到保存频率倍数
3. checkpoint目录有写权限

**Q: 如何解读H_res矩阵？**
A: 
- 对角线应接近1（保持残差流）
- 非对角线受gamma和tanh(logits)控制
- 非对角线大 -> 更多流混合
- 使用analyze_h_res_matrices.py进行详细分析
