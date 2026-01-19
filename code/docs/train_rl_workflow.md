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

## 1️⃣ 模块导入与全局配置 (Line 1-33)

### 功能说明
导入必要的库，并设置模型训练的关键配置参数。

```python
# 核心导入
from MySAC import config
from MySAC.preprocessors import FeatureEngineer, data_split
from MySAC.models.DRLAgent import DRLAgent
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env

# 全局配置参数
fix_seed = 1999                    # 随机种子
version = 'CSI/'                   # 数据集版本
model_name = 'StockFormer/'        # 模型名称
prediction_len = [1, 5]            # 预测周期 [短期1天, 长期5天]

# 预训练模型路径
short_prediction_model_path = 'Transformer/pretrained/csi/Short/checkpoint.pth'
long_prediction_model_path = 'Transformer/pretrained/csi/Long/checkpoint.pth'
mae_model_path = 'Transformer/pretrained/csi/mae/checkpoint.pth'
```

### 关键参数说明

| 参数 | 含义 |
|------|------|
| `fix_seed` | 随机种子，确保实验可复现 |
| `version` | 数据集版本 (CSI沪深300) |
| `model_name` | 模型名称 |
| `short_prediction_model_path` | 短期预测Transformer模型路径 |
| `long_prediction_model_path` | 长期预测Transformer模型路径 |
| `mae_model_path` | MAE (Masked AutoEncoder) 模型路径 |
| `prediction_len` | 预测周期 [1天, 5天] |

---

## 2️⃣ 目录初始化 (Line 36-41)

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

### 目录结构

```
项目根目录/
├── trained_models/      # 保存训练好的模型
├── tensorboard_log/     # TensorBoard日志
└── results/             # 训练/测试结果
```

---

## 3️⃣ 数据加载与预处理 (Line 44-96)

### 3.1 股票数据加载 (Line 44-56)

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
    
    # 生成收益率标签（用于监督学习信号）
    temp_df['label_short_term'] = temp_df['close'].pct_change(periods=prediction_len[0]).shift(periods=(-1*prediction_len[0]))
    temp_df['label_long_term'] = temp_df['close'].pct_change(periods=prediction_len[1]).shift(periods=(-1*prediction_len[1]))
    temp_df['tic'] = pd.Series([ticker]*len(temp_df))
    df = pd.concat((df, temp_df))

df = df.sort_values(by=['date','tic'])
```

**关键处理：**
- 生成短期收益标签 `label_short_term`: 1天收益率
- 生成长期收益标签 `label_long_term`: 5天收益率

### 3.2 技术指标生成 (Line 58-69)

```python
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=False,
    user_defined_feature=False
)

print("generate technical indicator...")
df = fe.preprocess_data(df)

# 重新整理索引
df = df.sort_values(['date','tic'], ignore_index=True)
df.index = df.date.factorize()[0]
```

**技术指标列表** (来自config):
- `macd` - 移动平均收敛散度
- `boll_ub` - 布林带上轨
- `boll_lb` - 布林带下轨
- `rsi_30` - 30日相对强弱指标
- `cci_30` - 30日顺势指标
- `dx_30` - 30日动向指标
- `close_30_sma` - 30日简单移动平均
- `close_60_sma` - 60日简单移动平均

### 3.3 协方差矩阵计算 (Line 71-88)

```python
cov_list = []
return_list = []

# 回看窗口为一年 (252个交易日)
lookback = 252
for i in range(lookback, len(df.index.unique())):
    data_lookback = df.loc[i-lookback:i, :]
    price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)
    
    # 计算协方差矩阵
    covs = return_lookback.cov().values
    cov_list.append(covs)

# 合并协方差数据
df_cov = pd.DataFrame({
    'date': df.date.unique()[lookback:],
    'cov_list': cov_list,
    'return_list': return_list
})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)
```

**作用：** 计算252天(约1年)回看窗口的股票收益协方差矩阵，用于捕捉股票间的相关性。

### 3.4 数据标准化 (Line 91-96)

```python
scaler = StandardScaler()
df_data = df[config.TECHNICAL_INDICATORS_LIST]

# 处理无穷值
df_data = df_data.replace([np.inf], config.INF)
df_data = df_data.replace([-np.inf], config.INF*(-1))

# 标准化技术指标
data = scaler.fit_transform(df_data.values)
df[config.TECHNICAL_INDICATORS_LIST] = data
```

**处理流程：**
1. 替换无穷值 → 2. StandardScaler标准化技术指标

---

## 4️⃣ 数据集划分 (Line 98-104)

```python
train = data_split(df, '2011-01-17', '2018-12-28')
eval = data_split(df, '2019-01-02', '2021-12-31')
test = data_split(df, '2018-10-09', '2022-04-16')

stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
```

### 数据划分时间线

```
训练集 Train:  2011-01-17 ───────────────────> 2018-12-28
验证集 Eval:                                    2019-01-02 ───> 2021-12-31
测试集 Test:                          2018-10-09 ─────────────────────> 2022-04-16
```

---

## 5️⃣ 环境参数配置 (Line 106-142)

```python
# GPU/CPU 设备选择
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# 环境配置参数
env_kwargs = {
    "hmax": 100,                    # 单只股票最大持仓数量
    "initial_amount": 100000,       # 初始资金
    "transaction_cost_pct": 0,      # 交易成本比例
    "state_space": state_space,     # 状态空间维度
    "stock_dim": stock_dimension,   # 股票数量
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "temporal_feature_list": config.TEMPORAL_FEATURE,
    "additional_list": config.ADDITIONAL_FEATURE,
    "action_space": stock_dimension,
    "reward_scaling": 100,          # 奖励缩放因子
    "figure_path": 'results/figures/'+version+model_name,
    "csv_path": 'results/csv/'+version+model_name,
    "mode": 'train',
    "time_window_start": config.time_window_start,
    "step_len": 500,                # 每个episode的最大步数
    "temporal_len": 60,             # 时间窗口长度 (60天)
    "hidden_channel": 128,          # Transformer隐藏层维度
    "model_name": model_name[:-1],
    "short_prediction_model_path": short_prediction_model_path,
    "long_prediction_model_path": long_prediction_model_path,
    "device": device,
}

# 模型和日志目录
model_dir = os.path.join(config.TRAINED_MODEL_DIR, version[:-1], model_name[:-1])
log_dir = os.path.join(config.RESULTS_DIR, version[:-1], model_name[:-1])
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
```

### 核心参数说明

| 参数 | 值 | 含义 |
|------|----|------|
| `hmax` | 100 | 单只股票最大持仓数量 |
| `initial_amount` | 100000 | 初始资金 |
| `transaction_cost_pct` | 0 | 交易成本比例 |
| `reward_scaling` | 100 | 奖励缩放因子 |
| `step_len` | 500 | 每个episode的最大步数 |
| `temporal_len` | 60 | 时间窗口长度 (60天) |
| `hidden_channel` | 128 | Transformer隐藏层维度 |

---

## 6️⃣ 训练流程 (Line 145-221)

### 6.1 流程控制

```python
print("Initial Env...")
train_mode = True  # 设置为True启用训练，False跳过训练直接测试
```

### 6.2 环境创建与包装 (Line 147-177)

```python
if train_mode:
    # 创建训练环境
    env_name = "train"
    env_kwargs["mode"] = env_name
    train_trade_gym = Env(df=train, **env_kwargs)
    env_train, _ = train_trade_gym.get_sb_env()

    # 创建验证环境
    env_name = "eval"
    env_kwargs["mode"] = env_name
    env_kwargs["time_window_start"] = [env_kwargs["temporal_len"]]  # 60
    eval_trade_gym = Env(df=eval, **env_kwargs)
    env_eval, _ = eval_trade_gym.get_sb_env()

    # 检查是否存在预训练模型
    load_pretrain = False
    final_model_path = os.path.join('trained_models/', version, model_name, 'best_train_model000.zip')
    if os.path.exists(final_model_path):
        load_pretrain = True
        
    # VecNormalize 包装 - 对观察值和奖励进行标准化
    vn_path = os.path.join('trained_models/', version, model_name, 'vec_normalize.pkl')
    if load_pretrain:
        if os.path.exists(vn_path):
            env_train_vn = VecNormalize.load(vn_path, env_train)
            env_eval_vn = VecNormalize.load(vn_path, env_eval)
            print(f"Loaded VecNormalize stats from {vn_path}")
        else:
            print(f"Can not loaded VecNormalize stats!!!")
    else:
        env_train_vn = VecNormalize(env_train, norm_reward=True, norm_obs=True)
        env_eval_vn = VecNormalize(env_eval, norm_reward=True, norm_obs=True)
    
    # VecMonitor 包装 - 日志监控
    env_train_vm = VecMonitor(env_train_vn, log_dir+'_train')
    env_eval_vm = VecMonitor(env_eval_vn, log_dir+'_test')
```

**环境包装层次：**

```
StockTradingEnv (原始环境)
    ↓ get_sb_env()
DummyVecEnv (向量化)
    ↓
VecNormalize (观察值/奖励标准化)
    ↓
VecMonitor (日志监控)
```

### 6.3 MAE-SAC模型参数 (Line 179-196)

```python
MAESAC_PARAMS = {
    "batch_size": 32,              # 批次大小
    "buffer_size": 100000,         # 经验回放缓冲区大小
    "learning_rate": 0.0001,       # 学习率
    "learning_starts": 100,        # 开始学习前的步数
    "ent_coef": "auto_0.1",        # 熵系数（自动调整）
    "enc_in": 96,                  # 编码器输入维度
    "dec_in": 96,                  # 解码器输入维度
    "c_out_construction": 96,      # 重构输出维度
    "d_model": 128,                # Transformer模型维度
    "d_ff": 256,                   # 前馈网络维度
    "n_heads": 4,                  # 注意力头数
    "e_layers": 2,                 # 编码器层数
    "d_layers": 1,                 # 解码器层数
    "dropout": 0.05,               # Dropout比例
    "transformer_path": mae_model_path,
    "transformer_device": device,
}
```

### 6.4 模型创建/加载与训练 (Line 198-221)

```python
# 创建DRL代理
agent = DRLAgent(env=env_train_vm)

# 加载预训练模型或创建新模型
if load_pretrain:
    print(f"load: {final_model_path}...")
    model_sac = SAC_MAE.load(final_model_path, env=env_train_vm, tensorboard_log=tensorboard_log_dir)
else:
    model_sac = agent.get_model("maesac", model_kwargs=MAESAC_PARAMS, 
                                 tensorboard_log=tensorboard_log_dir, seed=fix_seed)

# 生成带时间戳的日志名
timestamp = datetime.datetime.now().strftime("%H%M%S")
tb_log_name_with_timestamp = model_name[:-1] + '_' + timestamp + '/'

# 开始训练
print('Start training...')
start = time.time()
trained_sac = agent.train_model(
    model=model_sac,
    tb_log_name=tb_log_name_with_timestamp,
    check_freq=50000,              # 评估频率
    log_dir=log_dir,
    model_dir=model_dir,
    eval_env=env_eval_vm,
    total_timesteps=30000          # 总训练步数
)
end = time.time()
print("Training time: %.3f" % (end-start))

# 保存 VecNormalize 统计信息（用于测试时复用）
env_train_vn.save(os.path.join(model_dir, 'vec_normalize.pkl'))
```

**训练配置：**
- `check_freq`: 每50000步进行一次评估
- `total_timesteps`: 总训练步数30000

---

## 7️⃣ 测试流程 (Line 224-249)

```python
# 模型和标准化文件路径
model_path = os.path.join('trained_models/', version, model_name, 'best_train_model.zip')
vn_path = os.path.join('trained_models/', version, model_name, 'vec_normalize.pkl')

# 创建测试环境
env_name = "test"
env_kwargs["mode"] = env_name
env_kwargs["time_window_start"] = [env_kwargs["temporal_len"]]  # 60
test_trade_gym = Env(df=test, **env_kwargs)
env_test, _ = test_trade_gym.get_sb_env()

# 加载 VecNormalize 统计信息
if os.path.exists(vn_path):
    env_test_vn = VecNormalize.load(vn_path, env_test)
    print(f"Loaded VecNormalize from {vn_path}")
else:
    env_test_vn = VecNormalize(env_test, norm_reward=True, norm_obs=True)

# 执行测试预测
start = time.time()
results = DRLAgent.DRL_prediction_load_from_file(
    model_name='maesac',
    test_env=env_test_vn,
    cwd=model_path
)
end = time.time()
print("Test time: %.3f" % (end-start))

# 保存测试结果
df_root = 'results/df_print/' + version + model_name
os.makedirs(df_root, exist_ok=True)
assets_his, df_actions = results[1], results[2]
df_actions.to_csv(df_root + 'df_actions_test.csv')
assets_his.to_csv(df_root + 'df_assets_his_test.csv')
```

### 测试输出文件

```
results/df_print/CSI/StockFormer/
├── df_actions_test.csv    # 每步交易动作记录
└── df_assets_his_test.csv # 资产变化历史
```

---

## 📐 整体数据流图

```
                    ┌─────────────────────────────────────┐
                    │        原始股票CSV数据               │
                    │  (OHLCV + 指数数据)                  │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │       FeatureEngineer               │
                    │  (生成MACD/RSI/布林带等技术指标)      │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │       协方差矩阵计算                 │
                    │  (252天回看窗口)                     │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │       StandardScaler标准化          │
                    └─────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
┌─────────▼─────────┐     ┌───────────▼───────────┐    ┌──────────▼──────────┐
│   Train Data      │     │    Eval Data          │    │    Test Data        │
│ 2011-01~2018-12   │     │  2019-01~2021-12      │    │  2018-10~2022-04    │
└─────────┬─────────┘     └───────────┬───────────┘    └──────────┬──────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      StockTradingEnv (Gym环境)                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  状态空间: [协方差矩阵 + 技术指标 + Transformer隐藏特征 + 持仓量]    │    │
│  │  动作空间: [-1, 1]^stock_dim (连续动作，表示买卖比例)               │    │
│  │  奖励: 资产变化率 * reward_scaling                                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                    ┌─────────────────────▼─────────────────────┐
                    │            MAE-SAC Agent                  │
                    │  ┌─────────────────────────────────────┐  │
                    │  │  - 基于SAC (Soft Actor-Critic)       │  │
                    │  │  - 集成MAE Transformer特征提取       │  │
                    │  │  - 自动熵系数调整 (auto_0.1)         │  │
                    │  └─────────────────────────────────────┘  │
                    └─────────────────────┬─────────────────────┘
                                          │
                    ┌─────────────────────▼─────────────────────┐
                    │            输出结果                        │
                    │  - best_train_model.zip (最优模型)        │
                    │  - vec_normalize.pkl (标准化统计)         │
                    │  - df_actions_test.csv (交易动作)         │
                    │  - df_assets_his_test.csv (资产历史)      │
                    └───────────────────────────────────────────┘
```

---

## 🔧 核心组件依赖关系

| 组件 | 文件路径 | 功能 |
|------|----------|------|
| `config` | `MySAC/config.py` | 全局配置参数 |
| `DRLAgent` | `MySAC/models/DRLAgent.py` | 强化学习代理封装 |
| `SAC_MAE` | `MySAC/SAC/MAE_SAC.py` | 集成MAE的SAC算法 |
| `StockTradingEnv` | `envs/env_stocktrading_hybrid_control.py` | 股票交易Gym环境 |
| `FeatureEngineer` | `MySAC/preprocessors.py` | 特征工程处理器 |

---

## 📝 关键回调函数 (DRLAgent.py)

### EvalCallback
- 每隔 `check_freq` 步在验证集上评估模型
- 保存最优模型到 `model_dir`

### TensorboardCallback
- 记录训练过程中的指标到 TensorBoard

### oursTrainingRewardCallback
- 基于最近50个episode的平均奖励保存最优模型

---

## 🚀 运行方式

```bash
cd code
python train_rl.py
```

### 训练/测试模式切换

修改 `train_rl.py` 第145行：
```python
train_mode = True   # True: 训练+测试, False: 仅测试
```

---

## 📊 输出文件结构

```
code/
├── trained_models/
│   └── CSI/
│       └── StockFormer/
│           ├── best_train_model.zip    # 最优模型
│           └── vec_normalize.pkl       # 标准化统计
├── tensorboard_log/
│   └── mysac/
│       └── StockFormer_HHMMSS/         # TensorBoard日志
└── results/
    ├── CSI/
    │   └── StockFormer/
    │       └── evaluations.npz         # 评估结果
    └── df_print/
        └── CSI/
            └── StockFormer/
                ├── df_actions_test.csv    # 交易动作
                └── df_assets_his_test.csv # 资产历史
```
