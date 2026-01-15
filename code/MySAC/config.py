import random

version_name = 'CSI'#'N100'#
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

START_DATE = "2010-01-01"
END_DATE = "2022-05-07"

INF = 1100

## Model Parameters
MAESAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": 0.0,#"auto_0.1",
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

CSI_date = ['20110117', '20180801', '20180508', '20201231',  '20210104', '20220426']

date_dict = {'CSI': CSI_date, 'TEST': CSI_date}


# 使用随机种子动态生成随机窗口起始位置
# 范围 [60, 940] 对应 temporal_len=60 和 step_len=1000 的约束
# _rand.seed(1999)  # 注释掉固定种子，使用系统随机种子
time_window_start = [60] + [random.randint(60, 940) for _ in range(500)]