import os
import random
import torch

version_name = 'CSI'#'N100'#
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

START_DATE = "2010-01-01"
END_DATE = "2022-05-07"

fix_seed = 2022
INF = 1100

# 检测GPU可用性并决定使用GPU还是CPU
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

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
    "gamma": 0.9999,#折扣因子,越小越重视短期奖励，最大为1
    # "tau": 0.005,#目标网络软更新系数，越小更新越慢，一般取0.001-0.005
    "transformer_path":'',#mae_model_path,
    "transformer_device": device,
    "train_freq": 99,  # 每5步训练一次
    "gradient_steps": 99,  # 每次训练进行5个梯度更新
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

# Transformer Default Parameters
TRANSFORMER_PARAMS_DEFAULT = {
    # Model and experiment settings
    "model": "Transformer",                        # model of the experiment
    "project_name": "baseline",                   # name of the experiment

    # Data settings
    "data_name": "CSI",                           #
    "data_type": "stock",                         # stock
    "root_path": "data/",                         # root path of the data file
    "full_stock_path": "CSI/",                    # root path of the data file

    # Experiment type
    "exp_type": "pred",                           # [mae|pred]

    # Sequence lengths
    "seq_len": 60,                                # input series length
    "label_len": 1,                               # help series length
    "pred_len": 1,                                # predict series length

    # Model dimensions
    "enc_in": 96,                                 # encoder input size: cov+technical indicators
    "dec_in": 96,                                 # decoder input size
    "c_out": 96,                                  # output size[96|1](pred|mae)

    # Prediction settings
    "short_term_len": 1,                          # short term prediction len
    "long_term_len": 5,                           # long term prediction len
    "pred_type": "label_long_term",               # [label_long_term|label_short_term]
    "d_model": 128,                               # dimension of model
    "n_heads": 4,                                 # num of heads
    "e_layers": 2,                                # num of encoder layers
    "d_layers": 1,                                # num of decoder layers
    "d_ff": 256,                                  # dimension of fcn

    # Training settings
    "dropout": 0.05,                              # dropout
    "activation": "gelu",                         # activation
    "num_workers": 10,                            # data loader num workers
    "rank_alpha": 0.1,                            # weight of rank loss - adjust
    "itr": 2,                                     # each params run iteration
    "train_epochs": 3,                            # train epochs
    "batch_size": 32,                             # input data batch size
    "patience": 3,                                # early stopping patience
    "learning_rate": 0.0001,                      # optimizer learning rate
    "adjust_interval": 1,                         # lr adjust interval
    "des": "pred",                                # exp description
    "loss": "mse",                                # loss function
    "lradj": "type1",                             # adjust learning rate

    # GPU settings
    "use_gpu": True,                              # use gpu
    "gpu": 0,                                     # gpu
    "use_multi_gpu": False,                       # use multiple gpus
    "devices": "0,1,2,3",                         # device ids of multile gpus
}

# Transformer Prediction Parameters (short term)
TRANSFORMER_PARAMS_PRED_SHORT = {
    "project_name": "transformer_CSI_predShort",
    "exp_type": "pred",
    "train_epochs": 3,
    "itr": 1,
    "batch_size": 32,
    "seq_len": 60,
    "label_len": 1,
    "pred_len": 1,
    "enc_in": 10,
    "dec_in": 10,
    "c_out": 1,
    "d_model": 128,
    "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.05,
    "rank_alpha": 1.0,
    "learning_rate": 0.0001,
    "adjust_interval": 10,
    "num_workers": 10,
    "devices": 0,
    "short_term_len": 1,
    "long_term_len": 5,
    "pred_type": "label_short_term",
}

# Transformer Prediction Parameters (long term)
TRANSFORMER_PARAMS_PRED_LONG = {
    "project_name": "transformer_CSI_predLong",
    "exp_type": "pred",
    "train_epochs": 3,
    "itr": 1,
    "batch_size": 32,
    "seq_len": 60,
    "label_len": 1,
    "pred_len": 1,
    "enc_in": 10,
    "dec_in": 10,
    "c_out": 1,
    "d_model": 128,
    "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.05,
    "rank_alpha": 0.5,
    "learning_rate": 0.0001,
    "adjust_interval": 10,
    "num_workers": 10,
    "devices": 0,
    "short_term_len": 1,
    "long_term_len": 5,
    "pred_type": "label_long_term",
}

# Transformer MAE Parameters
TRANSFORMER_PARAMS_MAE = {
    "project_name": "transformer_CSI_mae",
    "exp_type": "mae",
    "train_epochs": 3,
    "itr": 1,
    "enc_in": 96,
    "dec_in": 96,
    "c_out": 96,
    "d_model": 128,
    "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.05,
    "learning_rate": 0.0001,
    "adjust_interval": 10,
    "num_workers": 10,
    "devices": 0,
    "short_term_len": 1,
    "long_term_len": 5,
}

# use CSI -300 ticker
# USE_CSI_300_TICKET = ['600519.SS',
#  '601318.SS',
#  '600036.SS',
#  '000858.SZ',
#  '600276.SS',
#  '601166.SS',
#  '601888.SS',
#  '300059.SZ',
#  '000651.SZ',
#  '600900.SS',
#  '600887.SS',
#  '000001.SZ',
#  '000725.SZ',
#  '600030.SS',
#  '300015.SZ',
#  '601398.SS',
#  '000568.SZ',
#  '600031.SS',
#  '600309.SS',
#  '000002.SZ',
#  '600809.SS',
#  '601919.SS',
#  '002142.SZ',
#  '600436.SS',
#  '601328.SS',
#  '601899.SS',
#  '002304.SZ',
#  '002352.SZ',
#  '002230.SZ',
#  '300014.SZ',
#  '600000.SS',
#  '600438.SS',
#  '600837.SS',
#  '000661.SZ',
#  '000100.SZ',
#  '000063.SZ',
#  '002241.SZ',
#  '002271.SZ',
#  '600585.SS',
#  '600690.SS',
#  '601601.SS',
#  '601668.SS',
#  '002027.SZ',
#  '600016.SS',
#  '600763.SS',
#  '600196.SS',
#  '000338.SZ',
#  '600048.SS',
#  '600703.SS',
#  '002129.SZ',
#  '600050.SS',
#  '601688.SS',
#  '600660.SS',
#  '600104.SS',
#  '600570.SS',
#  '601766.SS',
#  '601169.SS',
#  '600999.SS',
#  '002311.SZ',
#  '002371.SZ',
#  '600019.SS',
#  '002049.SZ',
#  '600406.SS',
#  '601088.SS',
#  '601988.SS',
#  '000538.SZ',
#  '000625.SZ',
#  '600745.SS',
#  '600028.SS',
#  '600893.SS',
#  '600346.SS',
#  '601628.SS',
#  '600588.SS',
#  '601009.SS',
#  '601390.SS',
#  '601857.SS',
#  '600009.SS',
#  '600132.SS',
#  '600584.SS',
#  '000776.SZ',
#  '000895.SZ',
#  '002001.SZ',
#  '600111.SS',
#  '600426.SS',
#  '601939.SS',
#  '000166.SZ',
#  '002050.SZ',
#  '002179.SZ']


USE_TICKET = os.listdir('data/'+ version_name)
USE_CSI_300_TICKET = [file.replace('.csv', '') for file in USE_TICKET]
use_ticker_dict = {'CSI':USE_CSI_300_TICKET, 'TEST': USE_CSI_300_TICKET[:5]}

CSI_date = ['2011-01-17','2018-12-28', '2019-01-02', '2021-12-30','2018-10-09', '2022-04-16']

CSI_date_trans = ['20110419', '20181228', '20180102', '20201231',  '20190402', '20211231']
date_dict = {'CSI': CSI_date_trans, 'TEST': CSI_date_trans}


step_len = 1000  # 每个训练/测试阶段的时间步长度
# 方案1：使用有序滑动窗口（Sliding Window）生成起始位置，减少环境突变
# 这种方式在 step_len 较小时能保证训练数据在宏观时间上的连续性
# 设定步长为 100，确保相邻 Episode 之间有数据重叠或紧密衔接
random.seed(fix_seed)
stride = int(step_len/5) #步长为 step_len 的五分之一
rand_start= random.randint(0, stride)
time_window_start = [i+rand_start for i in range(60, 2000 - stride * 4, stride)]