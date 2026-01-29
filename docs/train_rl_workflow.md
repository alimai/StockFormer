# 📊 `train_rl.py` 工作流程文档

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         train_rl.py                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 模块导入 & 全局配置                                          │
│  2. 目录初始化                                                   │
│  3. 数据加载与预处理                                             │
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
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env

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

## 3️⃣ 数据加载与预处理 (Line 46-96)

### 3.1 股票数据加载 (Line 46-56)

```python
# 创建空DataFrame
df = pd.DataFrame([], columns=['date','open','close','high','low','volume',
                               'dopen','dclose','dhigh','dlow','dvolume','price','tic'])

# 遍历所有股票，加载数据
for ticker in ticker_list:
    temp_df = pd.read_csv(os.path.join(full_stock_dir, ticker+'.csv'), 
                          usecols=['date', 'open', 'close', 'high', 'low', 'volume', 
                                   'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price'])
    temp_df['date'] = pd.to_datetime(temp_df['date'].apply(lambda x: str(x)))
    
    # 生成收益率标签（用于预测辅助任务）
    temp_df['label_short_term'] = temp_df['close'].pct_change(periods=prediction_len[0]).shift(periods=(-1*prediction_len[0]))
    temp_df['label_long_term'] = temp_df['close'].pct_change(periods=prediction_len[1]).shift(periods=(-1*prediction_len[1]))
    temp_df['tic'] = pd.Series([ticker]*len(temp_df))
    df = pd.concat((df, temp_df))

df = df.sort_values(by=['date','tic'])
```

### 3.2 技术指标生成 (Line 58-69)

```python
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=False,
    user_defined_feature=False
)
df = fe.preprocess_data(df)
```

### 3.3 协方差矩阵计算 (Line 71-88)

```python
# 回看窗口为一年 (252个交易日)
lookback = 252
for i in range(lookback, len(df.index.unique())):
    # 【修复】使用 i-1 排除当天数据，避免数据泄露
    data_lookback = df.loc[i-lookback:i-1, :]
    price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
    # ... 计算 covs ...
```

### 3.4 数据标准化 (Line 101-112)

```python
# 【优化】StandardScaler 只在训练数据上 fit，避免测试集信息泄露
scaler = StandardScaler()
train_mask = (df['date'] >= TRAIN_START) & (df['date'] < TRAIN_END)
train_data_for_scaler = df.loc[train_mask, config.TECHNICAL_INDICATORS_LIST]
scaler.fit(train_data_for_scaler.values)

# 对所有数据进行 transform
df[config.TECHNICAL_INDICATORS_LIST] = scaler.transform(df[config.TECHNICAL_INDICATORS_LIST].values)
```

---

## 4️⃣ 数据集划分 (Line 91-100)

使用 `config.CSI_date` 中定义的日期范围进行划分：

```python
TRAIN_START, TRAIN_END = config.CSI_date[0], config.CSI_date[1]
EVAL_START, EVAL_END = config.CSI_date[2], config.CSI_date[3]
TEST_START, TEST_END = config.CSI_date[4], config.CSI_date[5]

train = data_split(df, TRAIN_START, TRAIN_END)
eval = data_split(df, EVAL_START, EVAL_END)
test = data_split(df, TEST_START, TEST_END)
```

---

## 5️⃣ 环境参数配置 (Line 118-142)

```python
env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "transaction_cost_pct": 0,
    "reward_scaling": 100,
    "step_len": config.step_len,    # 步长迁移至 config
    "temporal_len": 60,
    "hidden_channel": 128,
    "short_prediction_model_path": short_prediction_model_path,
    "long_prediction_model_path": long_prediction_model_path,
    "device": config.device,
    # ... 其他参数 ...
}
```

---

## 6️⃣ 训练流程 (Line 147-214)

### 6.1 环境包装 (层次修复)

```python
# 【修复】调整包装顺序：先 VecMonitor 再 VecNormalize
# 这样 VecMonitor 记录原始奖励，VecNormalize 进行训练归一化
env_train_vm = VecMonitor(env_train, log_dir+'_train')
env_train_vn = VecNormalize(env_train_vm, norm_reward=True, norm_obs=True, ...)
```

### 6.2 模型创建与训练

```python
# 设置 MAE 预训练路径
config.MAESAC_PARAMS["transformer_path"] = mae_model_path
model_sac = agent.get_model("maesac", model_kwargs=config.MAESAC_PARAMS, ...)

# 开始训练
trained_sac = agent.train_model(
    model=model_sac,
    total_timesteps=30000,
    # ...
)
```

---

## 7️⃣ 测试流程 (Line 217-253)

### 7.1 测试环境设置

```python
# 加载训练时的 VecNormalize 统计信息
env_test_vn = VecNormalize.load(vn_path, env_test)
env_test_vn.training = False     # 冻结统计信息
env_test_vn.norm_reward = False  # 测试时不归一化奖励
```

### 7.2 结果保存 (Line 247-253)

测试结果保存至 `results/test/` 目录下。

```python
df_root = 'results/test/' + version + model_name
os.makedirs(df_root, exist_ok=True)
actions_test.to_csv(df_root + 'df_actions_test.csv')
assets_test.to_csv(df_root + 'df_assets_test.csv')
```

---

## 📐 整体数据流图 (更新版)

与原版基本一致，但关键更新点在于：
1. **配置中心化**：大部分超参数从代码硬编码转移到 `utils/config.py`。
2. **安全性修复**：协方差计算与标准化过程严格防止数据泄露。
3. **环境包装顺序**：`Monitor(Normalize(Env))` -> `Normalize(Monitor(Env))` 以获得正确的日志输出。