# 📊 `train_rl.py` 工作流程文档

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         train_rl.py                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 模块导入 & 全局配置                                          │
│  2. 目录初始化                                                   │
│  3. 统一数据加载与预处理 (Stock_Data)                            │
│  4. 数据集划分 (Train/Eval/Test)                                 │
│  5. 环境参数配置                                                 │
│  6. 训练流程 (可选)                                              │
│  7. 测试流程                                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ 模块导入与全局配置 (Line 1-45)

### 功能说明
导入必要的库，并设置模型训练的关键配置参数。核心配置已迁移至 `utils/config.py`。

```python
# 核心导入
from utils import config
from utils.preprocess import FeatureEngineer, data_split
from MySAC.models.DRLAgent import DRLAgent
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
from stable_baselines3.common.vec_env import VecMonitor
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env
from utils.data.stock_data_handle import Stock_Data

# 动态获取当前脚本路径
working_path = os.path.dirname(os.path.abspath(__file__))

# 全局配置参数 (部分由 config.py 提供)
version = 'CSI/'                   # 数据集版本
model_name = 'StockFormer/'        # 模型名称
prediction_len = [1, 5]            # 预测周期 [短期1天, 长期5天]

# 预训练模型路径 (基于 working_path)
short_prediction_model_path = working_path + '/Transformer/pretrained/csi/Short/checkpoint.pth'
long_prediction_model_path =  working_path + '/Transformer/pretrained/csi/Long/checkpoint.pth'
mae_model_path = working_path + '/Transformer/pretrained/csi/mae/checkpoint.pth'
full_stock_dir = 'data/CSI/'
ticker_list = config.use_ticker_dict['CSI']
```

### 关键配置 (utils/config.py)

| 参数 | 含义 |
|------|------|
| `fix_seed` | 随机种子 (e.g., 2022) |
| `device` | 计算设备 (cuda:0 或 cpu) |
| `TECHNICAL_INDICATORS_LIST` | 技术指标列表 (macd, boll, rsi, cci, dx, sma等) |
| `MAESAC_PARAMS` | MAE-SAC 训练超参数 |

---

## 2️⃣ 目录初始化 (Line 38-43)

### 功能说明
创建模型训练、日志记录、结果保存所需的目录结构。

```python
if not os.path.exists(config.TRAINED_MODEL_DIR):
    os.makedirs(config.TRAINED_MODEL_DIR)
if not os.path.exists(config.TENSORBOARD_LOG_DIR):
    os.makedirs(config.TENSORBOARD_LOG_DIR)
if not os.path.exists(config.RESULTS_DIR):
    os.makedirs(config.RESULTS_DIR)
```

---

## 3️⃣ 统一数据处理 (Stock_Data) (Line 46-60)

### 功能说明
使用 `Stock_Data` 类统一管理数据的加载、特征工程、协方差计算以及标准化。该类内部实现了严格的“防泄露”逻辑（如协方差计算使用 `i-1` 之前的数据，标准化仅在训练集上 `fit`）。

```python
# 初始化数据管理器
data_manager = Stock_Data(
    root_path='data/', 
    dataset_name='CSI', 
    full_stock_path='CSI/', 
    size=[60, 1, 1], 
    prediction_len=prediction_len
)
```

---

## 4️⃣ 数据集划分 (Line 62-64)

直接从 `data_manager` 获取预处理并划分好的 `DataFrame`：

```python
train = data_manager.get_split_df('train')
eval = data_manager.get_split_df('valid')
test = data_manager.get_split_df('test')
```

---

## 5️⃣ 环境参数配置 (Line 71-96)

```python
env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "transaction_cost_pct": 0,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "temporal_feature_list": config.TEMPORAL_FEATURE,
    "type_list": config.TYPE_FEATURE,
    "reward_scaling": 100,
    "step_len": config.step_len,
    "temporal_len": 60,
    "hidden_channel": 128,
    "short_prediction_model_path": short_prediction_model_path,
    "long_prediction_model_path": long_prediction_model_path,
    "device": config.device,
}
```

---

## 6️⃣ 训练流程 (Line 104-140)

### 6.1 环境包装

当前版本移除了 `VecNormalize`，直接使用 `VecMonitor` 记录日志。

```python
# 训练环境包装
train_trade_gym = Env(df = train, **env_kwargs)
env_train, _ = train_trade_gym.get_sb_env()
env_train_vm = VecMonitor(env_train, log_dir)

# 评估环境包装
eval_trade_gym = Env(df = eval, **env_kwargs)
env_eval, _ = eval_trade_gym.get_sb_env()
env_eval_vm = VecMonitor(env_eval, log_dir)
```

### 6.2 模型创建与训练

```python
# 设置 MAE 预训练路径并创建 Agent
config.MAESAC_PARAMS["transformer_path"] = mae_model_path
agent = DRLAgent(env = env_train_vm)
model_sac = agent.get_model("maesac", model_kwargs=config.MAESAC_PARAMS, ...)

# 开始训练
trained_sac = agent.train_model(
    model=model_sac,
    eval_env=env_eval_vm,
    total_timesteps=30000, # 示例步数
    ...
)
```

---

## 7️⃣ 测试流程 (Line 143-162)

### 7.1 测试环境设置

测试阶段使用原始环境，不进行额外的包装。

```python
test_trade_gym = Env(df = test, **env_kwargs)
env_test, _ = test_trade_gym.get_sb_env()
```

### 7.2 结果加载与保存

```python
# 从文件加载训练好的模型进行预测
model_path = os.path.join('trained_models/', version, model_name, 'best_train_model.zip')
results = DRLAgent.DRL_prediction_load_from_file(model_name='maesac', test_env=env_test, cwd=model_path)

# 保存资产曲线和动作序列
assets_test, actions_test = results[1], results[2]
actions_test.to_csv(df_root + 'df_actions_test.csv')
assets_test.to_csv(df_root + 'df_assets_test.csv')
```

---

## 📐 整体数据流图 (简化版)

```
[原始CSV] -> [Stock_Data 统一预处理] -> [Train/Eval/Test DF]
                                          |
                    ┌─────────────────────┴─────────────────────┐
                    v                                           v
            [Env (Training)]                            [Env (Testing)]
                    |                                           |
            [VecMonitor]                                [DRL Prediction]
                    |                                           |
            [MAE-SAC Model]                             [Results (CSV)]
```

**关键更新点**：
1. **组件解耦**：移除 `VecNormalize` 简化了环境链，避免了测试集均值/方差漂移问题。
2. **中心化预处理**：`Stock_Data` 替代了零散的 `FeatureEngineer` 和手动协方差计算。
3. **加载机制**：测试阶段统一采用 `DRL_prediction_load_from_file` 接口。
