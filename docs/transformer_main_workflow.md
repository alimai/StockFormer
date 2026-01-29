# 📊 `transformer_main.py` 工作流程文档

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                       transformer_main.py                        │
├─────────────────────────────────────────────────────────────────┤
│  1. 参数解析 & 默认配置 (Args类)                                  │
│  2. 环境初始化 (随机种子, GPU配置)                               │
│  3. 数据加载与预处理 (Stock_Data)                                │
│  4. 实验选择 (Exp_pred 或 Exp_mae)                               │
│  5. 训练与测试流程 (exp.train, exp.test)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ 参数配置与初始化 (Line 1-45)

### 功能说明
加载全局配置并合并特定的 Transformer 参数。

```python
# 核心导入
from Transformer.exp.exp_pred import Exp_pred
from Transformer.exp.exp_mae import Exp_mae
from utils.data.stock_data_handle import Stock_Data
from utils.config import TRANSFORMER_PARAMS_DEFAULT, TRANSFORMER_PARAMS_MAE, ...

# 参数合并逻辑
TRANSFORMER_PARAMS_TARGET = TRANSFORMER_PARAMS_MAE  # 可切换为 PRED_SHORT/LONG
args = Args({**TRANSFORMER_PARAMS_DEFAULT, **TRANSFORMER_PARAMS_TARGET})
```

### 关键配置源 (utils/config.py)
- `TRANSFORMER_PARAMS_DEFAULT`: 基础模型维度、训练周期、优化器设置等。
- `TRANSFORMER_PARAMS_MAE`: 关系表示学习相关的特定维度和设置。
- `TRANSFORMER_PARAMS_PRED_SHORT/LONG`: 收益率预测任务的特定设置。

---

## 2️⃣ 数据处理 (Stock_Data)

### 2.1 原始数据读取
- 遍历 `ticker_list` 加载 CSV。
- 生成 `label_short_term` (1天) 和 `label_long_term` (5天) 收益标签。

### 2.2 特征工程
- 使用 `FeatureEngineer` 生成技术指标 (MACD, RSI等)。
- **协方差矩阵计算**: 计算 252 天回看窗口的收益率协方差。
  - *注意*: 代码中包含修复逻辑，确保使用 `i-1` 之前的历史数据，防止数据泄露。

### 2.3 数据集划分
- 依据 `config.date_dict` 将日期划分为 `train`, `valid`, `test` 三个阶段。

---

## 3️⃣ 实验执行流程

根据 `args.exp_type` 选择不同的实验类：

### 3.1 掩码自编码器任务 (MAE) - `Exp_mae`
- **目标**: 学习股票间的关系表示。
- **机制**: 
  - 输入特征包含协方差矩阵和技术指标。
  - 在 `vali` 或 `train` 过程中对输入进行 Masking。
  - 模型尝试重建被掩码的部分。
- **损失函数**: MSELoss。

### 3.2 预测任务 (Pred) - `Exp_pred`
- **目标**: 预测短期或长期收益率。
- **机制**:
  - 输入当前状态特征。
  - 输出对应的收益率预测。
- **损失函数**: `MSELoss` + `RankingLoss` (通过 `rank_alpha` 控制权重)。

---

## 4️⃣ 核心组件细节

### 4.1 Transformer 模型 (Transformer_base)
- 位于 `code/Transformer/models/transformer.py`。
- 标准的 Encoder-Decoder 架构或仅 Encoder 架构 (取决于配置)。
- 处理维度: `d_model`, `n_heads`, `e_layers`, `d_layers` 等。

### 4.2 训练监控
- 使用 `tensorboardX.SummaryWriter` 记录 Loss 和指标。
- 采用 `EarlyStopping` 防止过拟合。
- `adjust_learning_rate` 动态调整学习率。

---

## 5️⃣ 运行与输出

### 运行方式
```bash
python transformer_main.py
```
*(注意: 需在代码中手动切换 `TRANSFORMER_PARAMS_TARGET` 指向 MAE 或 Pred 配置)*

### 输出内容
- **日志**: 存储于 `log/` 目录，可用 TensorBoard 查看。
- **模型检查点**: 训练完成后保存至 `checkpoints/` 目录。
- **测试结果**: 包括 MSE, MAE 以及 Ranking 相关指标。

---

## 📐 数据流图 (Transformer 专用)

```
[原始CSV] -> [FeatureEngineer] -> [Covariance Calculation]
                                          |
                                          v
                              [StandardScaler (Fit on Train)]
                                          |
                    ┌─────────────────────┴─────────────────────┐
                    v                                           v
            [Exp_mae (MAE)]                            [Exp_pred (Pred)]
      (Learn Representations)                    (Predict Returns)
    Masked Input -> Reconstruct                Input -> Return Prediction
             |                                          |
             v                                          v
      [Model Checkpoint]                         [Model Checkpoint]
```
