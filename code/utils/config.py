import os
import torch
from datetime import datetime

fix_seed = 2022
INF = 1100
SCALE_A = 1.2

# 检测GPU可用性并决定使用GPU还是CPU
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

version_name = 'CSI_2'#'N100'#
model_name='StockFormer'
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "log"
RESULTS_DIR = "results"

START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")#"2025-12-31"

USE_TICKET = os.listdir('data/'+ version_name)
USE_CSI_300_TICKET = [file.replace('.csv', '') for file in USE_TICKET]
if len(USE_CSI_300_TICKET) > 88:#如果大于88，取前88个
    USE_CSI_300_TICKET = USE_CSI_300_TICKET[:88]

#`train`, `valid`, `test` 三个阶段
#CSI_date_trans = ['20110419', '20181228', '20190712', '20220415',  '20181009', '20220415']
CSI_date_trans = ['20110419', '20220415', '20220630', '20250331',  '20220630', '20251231']

step_len = 800  # 每个Episode的时间步长度
stride = int(step_len * 0.6) # 滑动窗口步长,<step_len，确保相邻Episode之间有数据重叠

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

TYPE_FEATURE = [
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

TICKET_SIZE = len(USE_CSI_300_TICKET)
INDICATORS_SIZE = len(TECHNICAL_INDICATORS_LIST)
TEMPORAL_FEATURE_SIZE = len(TEMPORAL_FEATURE)
ENCODER_INPUT_SIZE = TICKET_SIZE + INDICATORS_SIZE

##transformer Model Parameters
MAESAC_PARAMS = {
    "batch_size": 128,#important,同时影响速度
    "buffer_size": 50000,
    "learning_rate": 0.0001,#除MAE模型外其他模块的学习率
    "learning_starts": 1000,
    "ent_coef": 0.001,#"auto_0.1",#key--同时影响actor_loss/critic_loss
    "enc_in": ENCODER_INPUT_SIZE,#MAE编码器的输入维度#股票数88+技术指标数8
    "dec_in": ENCODER_INPUT_SIZE,#MAE解码器的输入维度
    "c_out_construction": ENCODER_INPUT_SIZE,#MAE模型的输出维度（只用来评估重建损失）
    "d_model":128,#MAE模型的隐藏层维度（编码后，解码前，输入给SAC模型）
    "d_ff":256,#demension of Feed-Forward Network(FFN,前馈神经网络) in SAC Transformer,位于SAC编码/解码block内
    "n_heads":4,#多头注意力机制的头数
    "e_layers":2,#编码器层数
    "d_layers":1,#解码器层数
    "dropout":0.05,
    "gamma": 0.99,#折扣因子,越小越重视短期奖励，最大为1
    "transformer_path":'',#mae_model_path,
    "transformer_device": device,
    "train_freq": 249,  # 每x步训练一次
    "gradient_steps": 100,  # 每次训练进行x个梯度更新
}

# MAESAC_PARAMS_PRED = {
#     "batch_size": 128,
#     "buffer_size": 50000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1",
#     "enc_in":TEMPORAL_FEATURE_SIZE,#时序特征数10
#     "dec_in":TEMPORAL_FEATURE_SIZE,
#     "c_out_prediction":1,#不同于MAESAC_PARAMS
#     "d_model":128,
#     "d_ff":256,
#     "n_heads":8,#不同于MAESAC_PARAMS
#     "e_layers":3,#不同于MAESAC_PARAMS
#     "d_layers":2,#不同于MAESAC_PARAMS
#     "dropout":0.05,
#     "pred_len":1,#不同于MAESAC_PARAMS
#     "seq_len":60,#不同于MAESAC_PARAMS
# }



# Transformer Default Parameters
TRANSFORMER_PARAMS_DEFAULT = {
    # Model and experiment settings
    "model": "Transformer",                       # model of the experiment
    "project_name": "baseline",                   # name of the experiment
    "exp_type": "mae",                            # [mae|pred]


    # Data settings
    "data_name": version_name,                    #
    "data_type": "stock",                         # stock|crypto
    "full_stock_path": "data/"+version_name+"/",  # root path of the data file

    # Sequence lengths
    "seq_len": 60,                                # input series length
    "label_len": 1,                               # help series length
    "pred_len": 1,                                # predict series length

    # Model dimensions
    "enc_in": 96,                                 # encoder input size: cov+technical indicators
    "dec_in": 96,                                 # decoder input size
    "c_out": 96,                                  # output size[96|1](pred|mae)

    # Prediction settings
    "pred_type": "label_long_term",               # [label_long_term|label_short_term]
    "short_term_len": 1,                          # short term prediction len
    "long_term_len": 5,                           # long term prediction len
    "d_model": 128,                               # dimension of model
    "n_heads": 4,                                 # num of heads
    "e_layers": 2,                                # num of encoder layers
    "d_layers": 1,                                # num of decoder layers
    "d_ff": 256,                                  # dimension of fcn

    # Training settings
    "dropout": 0.05,                              # dropout
    "activation": "gelu",                         # activation
    "num_workers": 10,                             # data loader num workers
    "rank_alpha": 0.1,                            # weight of rank loss - adjust
    "itr": 2,                                     # each params run iteration
    "train_epochs": 3,                            # train epochs
    "batch_size": 64,                             # input data batch size
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
    "train_epochs": 5,
    "itr": 1,
    "batch_size": 32,
    "seq_len": 60,
    "label_len": 1,
    "pred_len": 1,
    "enc_in": TEMPORAL_FEATURE_SIZE,#时序指标10个
    "dec_in": TEMPORAL_FEATURE_SIZE,
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
    "train_epochs": 5,
    "itr": 1,
    "batch_size": 32,
    "seq_len": 60,
    "label_len": 1,
    "pred_len": 1,
    "enc_in": TEMPORAL_FEATURE_SIZE,#时序指标10个
    "dec_in": TEMPORAL_FEATURE_SIZE,
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
    "train_epochs": 30,
    "itr": 1,
    "enc_in": ENCODER_INPUT_SIZE,#编码器输入维度（股票数88+技术指标数8）
    "dec_in": ENCODER_INPUT_SIZE,#解码器输入维度（股票数88+技术指标数8）
    "c_out": ENCODER_INPUT_SIZE,#输出维度（股票数88+技术指标数8）
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
CSI_300_TICKET_download = ['600519.SS',
 '601318.SS',
 '600036.SS',
 '000858.SZ',
 '600276.SS',
 '601166.SS',
 '601888.SS',
 '300059.SZ',
 '000651.SZ',
 '600900.SS',
 '600887.SS',
 '000001.SZ',
 '000725.SZ',
 '600030.SS',
 '300015.SZ',
 '601398.SS',
 '000568.SZ',
 '600031.SS',
 '600309.SS',
 '000002.SZ',
 '600809.SS',
 '601919.SS',
 '002142.SZ',
 '600436.SS',
 '601328.SS',
 '601899.SS',
 '002304.SZ',
 '002352.SZ',
 '002230.SZ',
 '300014.SZ',
 '600000.SS',
 '600438.SS',
 '000661.SZ',
 '000100.SZ',
 '000063.SZ',
 '002241.SZ',
 '002271.SZ',
 '600585.SS',
 '600690.SS',
 '601601.SS',
 '601668.SS',
 '002027.SZ',
 '600016.SS',
 '600763.SS',
 '600196.SS',
 '000338.SZ',
 '600048.SS',
 '600703.SS',
 '002129.SZ',
 '600050.SS',
 '601688.SS',
 '600660.SS',
 '600104.SS',
 '600570.SS',
 '601766.SS',
 '601169.SS',
 '600999.SS',
 '002311.SZ',
 '002371.SZ',
 '600019.SS',
 '002049.SZ',
 '600406.SS',
 '601088.SS',
 '601988.SS',
 '000538.SZ',
 '000625.SZ',
 '600745.SS',
 '600028.SS',
 '600893.SS',
 '600346.SS',
 '601628.SS',
 '600588.SS',
 '601009.SS',
 '601390.SS',
 '601857.SS',
 '600009.SS',
 '600132.SS',
 '600584.SS',
 '000776.SZ',
 '000895.SZ',
 '002001.SZ',
 '600111.SS',
 '600426.SS',
 '601939.SS',
 '000166.SZ',
 '002050.SZ',
 '002179.SZ',
 '601018.SS']
