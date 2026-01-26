
import os
import random
import torch

version_name = 'CSI'#'N100'#
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

START_DATE = "2010-01-01"
END_DATE = "2022-05-07"

fix_seed = 1999
INF = 1100

# 检测GPU可用性并决定使用GPU还是CPU
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

## Model Parameters
MAESAC_PARAMS = {
    "batch_size": 128,#important
    "buffer_size": 10000,
    "learning_rate": 0.00001,
    "learning_starts": 100,
    "ent_coef": "auto_0.001",#key
    "enc_in": 96,
    "dec_in": 96,
    "c_out_construction": 96,
    "d_model":128,
    "d_ff":256,
    "n_heads":4,
    "e_layers":2,
    "d_layers":1,
    "dropout":0.05,
    "transformer_path":'',#mae_model_path,
    "transformer_device": device,
    "train_freq": 5,  # 每5步训练一次
    "gradient_steps": 5,  # 每次训练进行5个梯度更新
}

MAESAC_PARAMS_PRED = {
    "batch_size": 128,
    "buffer_size": 50000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
    "enc_in":108,
    "dec_in":108,
    "c_out_prediction":1,
    "d_model":128,
    "n_heads":8,
    "e_layers":3,
    "d_layers":2,
    "d_ff":256,
    "dropout":0.05,
    "pred_len":1,
    "seq_len":60,
}

ADDITIONAL_FEATURE = [
    'label_short_term',
    'label_long_term'
]

TEMPORAL_FEATURE = [
    'open', 
    'close', 
    'high', 
    'low', 
    'volume', 
    'dopen', 
    'dclose', 
    'dhigh', 
    'dlow', 
    'dvolume'
]

NORMALIZED_TEMPORAL_FEATURE = [
    'open', 
    'close', 
    'high', 
    'low', 
    'volume'
]

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    # "return_ratio",
]

USE_TICKET = os.listdir('../data/'+ version_name)
USE_CSI_300_TICKET = [file.replace('.csv', '') for file in USE_TICKET]
use_ticker_dict = {'CSI':USE_CSI_300_TICKET, 'TEST': USE_CSI_300_TICKET[:5]}

CSI_date = ['2011-01-17','2018-12-28', '2019-01-02', '2021-12-30','2018-10-09', '2022-04-16']

# 使用随机种子动态生成随机窗口起始位置
# 范围 [60, 940] 对应 temporal_len=60 和 step_len=1000 的约束
random.seed(fix_seed)
time_window_start = [60] + [random.randint(60, 1540) for _ in range(500)]  # 500个随机起始位置