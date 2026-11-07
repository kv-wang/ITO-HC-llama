# 代码修改总结

## 修改的文件

### 1. torchtitan/config_manager.py
添加了新的配置参数：
```python
--model.save_h_res_freq <int>  # 每N步保存一次H_res矩阵，默认0（禁用）
```

### 2. train.py
添加了两个新函数和相应的调用逻辑：

#### 新增函数

**`collect_gamma_values(model, model_config)`**
- 功能：收集所有层的gamma值
- 返回：字典，键为 `gamma/layer_{id}_{attn|ffn}`，值为gamma值
- 只在 `identity_tanh_offdiag` 方法下工作

**`save_h_res_matrices(model, model_config, step, save_dir)`**
- 功能：保存所有层的H_res矩阵到JSON文件
- 保存路径：`<save_dir>/h_res_step_{step}/h_res_matrices.json`
- 只在 `identity_tanh_offdiag` 方法下工作

#### 调用点

1. **初始化阶段**（约380行）：
   - 检查是否启用H_res保存
   - 创建保存目录

2. **日志记录阶段**（约740-765行）：
   - 每个log_freq步收集gamma值
   - 添加到wandb日志字典
   - 记录到wandb

3. **保存阶段**（约760-765行）：
   - 根据save_h_res_freq保存H_res矩阵
   - 只在rank 0执行

## 使用方法

### 基本用法

```bash
# 启用gamma记录（自动，只需启用wandb）
--metrics.enable_wandb

# 启用H_res矩阵保存（每100步）
--model.save_h_res_freq 100
```

### 完整示例

```bash
# 方法1: 修改run_llama_2_7b_hc_adamw.sh
METHODS="identity_tanh_offdiag" \
    ./run_llama_2_7b_hc.sh \
    --optimizer.name AdamW \
    --metrics.enable_wandb \
    --model.save_h_res_freq 100 \
    --training.steps 1000

# 方法2: 使用提供的示例脚本
./example_train_with_h_res_logging.sh
```

## 新增文件

1. **H_RES_LOGGING_README.md** - 详细的功能说明文档
2. **example_train_with_h_res_logging.sh** - 示例训练脚本
3. **analyze_h_res_matrices.py** - H_res矩阵分析工具
4. **test_h_res_logging.py** - 功能测试脚本

## 输出说明

### Wandb输出
在wandb界面中可以看到以下新增指标：
- `gamma/layer_0_attn` - 第0层attention的gamma值
- `gamma/layer_0_ffn` - 第0层FFN的gamma值
- `gamma/layer_1_attn` - 第1层attention的gamma值
- ... (每层都有)

### 本地文件输出
```
<checkpoint_folder>/
  h_res_matrices/
    h_res_step_100/
      h_res_matrices.json
    h_res_step_200/
      h_res_matrices.json
    ...
```

每个JSON文件包含：
```json
{
  "layer_0_attn": [[1.0, 0.01, ...], [0.01, 1.0, ...], ...],
  "layer_0_ffn": [[1.0, 0.02, ...], [0.02, 1.0, ...], ...],
  ...
}
```

## 查看结果

### 查看gamma演化
```bash
# 在wandb网站上查看
# 或使用wandb API
import wandb
api = wandb.Api()
run = api.run("your-project/your-run-id")
history = run.history()
print(history[['step', 'gamma/layer_0_attn', 'gamma/layer_0_ffn']])
```

### 分析H_res矩阵
```bash
# 使用提供的分析脚本
python analyze_h_res_matrices.py ./outputs/checkpoints/h_res_matrices/

# 或自己编写分析代码
import json
with open('path/to/h_res_step_100/h_res_matrices.json') as f:
    h_res = json.load(f)
    print(h_res['layer_0_attn'])
```

## 性能影响

- **Gamma记录**: 几乎无影响（每log_freq步读取几个参数）
- **H_res保存**: 
  - 只在指定步数执行
  - IO时间 < 1秒（矩阵很小，默认4x4）
  - 建议频率：100-1000步

## 验证

可以通过以下方式验证功能是否正常：

1. **检查wandb**：确认gamma指标出现
2. **检查文件**：确认h_res_matrices目录下有保存的JSON文件
3. **运行分析脚本**：`python analyze_h_res_matrices.py <dir>`

## 注意事项

1. 只支持 `identity_tanh_offdiag` 方法
2. H_res保存只在rank 0执行（避免重复）
3. 确保checkpoint目录有写权限
4. 磁盘空间考虑：每步约几KB，合理设置保存频率
