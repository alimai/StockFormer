
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

##transformer Model Parameters
MAESAC_PARAMS = {
    "batch_size": 32,#important
    "buffer_size": 10000,
    "learning_rate": 0.00001,
    "learning_starts": 100,
    "ent_coef": "auto_0.001",#key
    "enc_in": 96,#编码器的输入维度#股票数88+技术指标数8
    "dec_in": 96,#解码器的输入维度
    "c_out_construction": 96,#模型的输出维度
    "d_model":128,#模型的隐藏层维度
    "d_ff":256,#前馈神经网络的维度
    "n_heads":4,#多头注意力机制的头数
    "e_layers":2,#编码器层数
    "d_layers":1,#解码器层数
    "dropout":0.01,
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
    "enc_in":108,#股票数88+技术指标数8+时间特征数12???##股票数100+技术指标数8???
    "dec_in":108,
    "c_out_prediction":1,#不同于MAESAC_PARAMS
    "d_model":128,
    "d_ff":256,
    "n_heads":8,#不同于MAESAC_PARAMS
    "e_layers":3,#不同于MAESAC_PARAMS
    "d_layers":2,#不同于MAESAC_PARAMS
    "dropout":0.05,
    "pred_len":1,#不同于MAESAC_PARAMS
    "seq_len":60,#不同于MAESAC_PARAMS
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

step_len = 1000  # 每个训练/测试阶段的时间步长度
# 方案1：使用有序滑动窗口（Sliding Window）生成起始位置，减少环境突变
# 这种方式在 step_len 较小时能保证训练数据在宏观时间上的连续性
# 设定步长为 100，确保相邻 Episode 之间有数据重叠或紧密衔接
random.seed(fix_seed)
stride = int(step_len/5) #步长为 step_len 的五分之一
rand_start= random.randint(0, stride)
time_window_start = [i+rand_start for i in range(60, 1800-stride, stride)]