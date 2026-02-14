# 📊 `train_rl.py` 工作流程文档 (v2026.02 更新版)

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         train_rl.py                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 模块导入 & 全局配置 (Gymnasium 兼容)                         │
│  2. 目录初始化                                                   │
│  3. 统一数据加载与预处理 (Stock_Data)                            │
│  4. 数据集划分 (Train/Eval/Test)                                 │
│  5. 环境参数配置 (StockTradingEnv)                               │
│  6. 训练流程 (多重 Callback 监控)                                │
│  7. 测试流程 (模型加载预测)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ 依赖库升级与全局配置

### 核心变更：Gymnasium 迁移
本工程已全面从 `gym` 迁移至 `gymnasium`。主要变动包括：
- 导入路径：`import gymnasium as gym`。
- `step()` 函数返回 5 个值：`obs, reward, terminated, truncated, info`。
- `reset()` 函数返回 2 个值：`obs, info`。

### 核心导入
```python
from utils import config
from MySAC.models.DRLAgent import DRLAgent
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
from stable_baselines3.common.vec_env import VecMonitor
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env
from utils.data.stock_data_handle import Stock_Data
```

---

## 2️⃣ 统一数据处理 (Stock_Data)

使用 `Stock_Data` 类统一管理数据的加载、特征工程、协方差计算以及标准化。该类确保了协方差计算使用 `i-1` 之前的数据，并严格在训练集上 `fit` 标准化参数，防止数据泄露。

```python
data_manager = Stock_Data(
    root_path='data/', 
    dataset_name=version_name, 
    full_stock_path=version_name+'/', 
    size=[60, 1, 1], 
    prediction_len=[1, 5]
)
```

---

## 3️⃣ 训练环境与监控链

### 状态空间 (State Space) 优化 (v2026.02)
为了加速训练和减少内存占用，自 v2026.02 起，状态空间移除了冗余的隐藏层特征占位符。
- **旧版**: `[Covariances, Technical Indicators, Hidden_NP1, Hidden_NP2, Date_Features]`
- **新版**: `[Covariances, Technical Indicators, Date_Features]`
- **变更影响**: `Hidden_NP1/2` (伪随机噪声) 被移除，`MAE_SAC` 内部也不再对其进行切片提取，显著降低了内存带宽压力。

训练环境采用多层包装以确保日志记录和维度适配：
1. **StockTradingEnv (gymnasium.Env)**: 基础交易环境。
2. **DummyVecEnv**: 将单环境包装为向量化环境，满足 SB3 算法输入要求。
3. **VecMonitor**: 记录 `episode_reward` 和 `episode_length`。**关键点**：它会在 episode 结束时向 `info` 字典注入 `"episode"` 键，这是本工程唯一可靠的 episode 结束信号。

```python
train_trade_gym = Env(df = train, **env_kwargs)
env_train, _ = train_trade_gym.get_sb_env() # 返回 DummyVecEnv
env_train_vm = VecMonitor(env_train, log_path_train)
```

---

## 4️⃣ 回调系统 (Callback System)

`DRLAgent.train_model` 集成了三个互补的回调函数，用于全方位监控训练过程：

### 4.1 FinancialEvalCallback (继承自 EvalCallback)
- **触发频率**: `check_freq` (步数)。
- **职责**: 在评估环境中手动运行 `n_eval_episodes`，收集并记录 `reward_ratio` 和 `sharpe_ratio` 等金融指标到 TensorBoard。

### 4.2 CombinedCallback
- **职责**: 
  - **模型备份**: 每 2 个 episode 保存一次 `tmp_mode.zip`。
  - **训练奖励监控**: 定时加载日志，记录过去 10 个 episode 的平均奖励。
  - **最佳模型保存**: 当 `mean_reward` 创新高时，保存 `best_train_model.zip`。
- **唯一计数机制**: 放弃不稳定的 `done/dones` 布尔值，改为检查 `info` 中是否存在 `"episode"` 键。这确保了 `episode_count` 增加 1 的操作在每个 episode 结束时**有且仅触发一次**。

### 4.3 FinancialMetricsCallback
- **职责**: 在训练过程中实时捕获环境返回的金融指标（如步奖励、夏普率等）并写入 TensorBoard。

---

## 5️⃣ 模型训练与评估

```python
agent = DRLAgent(env = env_train_vm)
# 获取并训练模型
trained_sac = agent.train_model(
    model=model_sac,
    check_freq=2000,
    eval_env=env_eval_vm,
    total_timesteps=30000
)
```

---

## 6️⃣ 测试流程

测试阶段不使用包装器，直接通过 `DRL_prediction_load_from_file` 静态方法加载保存好的 `.zip` 模型，在测试集上运行并输出 CSV 结果。

```python
results = DRLAgent.DRL_prediction_load_from_file(
    model_name='maesac',
    test_env=env_test, 
    cwd=best_model_path
)
```

---
*文档最后更新：2026年2月11日 (基于重大依赖升级与回调系统重构)*
