import os
import sys
import random
import numpy as np
import torch
import argparse
from datetime import datetime

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='StockFormer Training Configuration')
    parser.add_argument('--struct_type', type=int, default=1,
                        help='结构类型1/2/3/4')
    parser.add_argument('--scale_ratio', type=int, default=1,
                        help='缩放比例')
    parser.add_argument('--seed', type=int, default=random.randint(1, 5000), # 2022 #
                        help='固定随机种子')
    # parser.add_argument('--struct_base_flag', type=str, default='True',
    #                     help='policy模式 (True/False)')
    args, unknown = parser.parse_known_args()
    return args

_args = parse_args() #在模块被导入时立即执行
struct_type = _args.struct_type
scale_ratio = _args.scale_ratio
fix_seed = _args.seed
#struct_base_flag = _args.struct_base_flag.lower() in ('true', '1', 'yes', 't')
ratio_max = 0.1

def set_seed(seed=fix_seed):
    """统一设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 检测 GPU 可用性并决定使用 GPU 还是 CPU
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

INF = 1100
SCALE_A = 1.2

version_name = 'CSI_2'#'N100'#
model_name='StockFormer'
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "log"
RESULTS_DIR = "results"

START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")#"2025-12-31"

USE_TICKET = os.listdir('data/'+ version_name)
USE_CSI_300_TICKET = [file.replace('.csv', '') for file in USE_TICKET]
MAX_TICKET_NUM = 88 # CSI 300 中的股票数量
if len(USE_CSI_300_TICKET) > MAX_TICKET_NUM:#如果大于 88，取前 88 个
    USE_CSI_300_TICKET = USE_CSI_300_TICKET[:MAX_TICKET_NUM]

#'train', 'valid', 'test' 三个阶段
#CSI_date_trans = ['20110419', '20181228', '20190712', '20220415',  '20181009', '20220415']
CSI_date_trans = ['20110419', '20220415', '20220630', '20250331',  '20220630', '20251231']

step_len = 800  # 每个 Episode 的时间步长度
stride = int(step_len * 0.6) # 滑动窗口步长，<step_len，确保相邻 Episode 之间有数据重叠

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
# ===== 可调节的超参数（加载预训练模型时可修改） =====
MAESAC_TUNABLE_PARAMS = {
    "batch_size": 128,
    "buffer_size": 80000,#用于存储环境的"经验"(Obs, Action, Reward, Next_Obs, Done)
    "buffer_max_load": 10000,# 【新增】指定从旧 Buffer 加载的数据条数，None 为全部加载
    "learning_starts": 1000,
    "learning_rate": 1e-4, # 1e-4,#所有模块学习率 #LinearSchedule(start=1e-4, end=1e-5, end_fraction=1.0),
    "ent_coef": "auto_0.01", #"auto_0.003",# 0.001#熵系数，key

    # 训练频率与折扣优化
    "train_freq": 5,#每 * 步训练一次，更新目标网络一次
    "gradient_steps": 1,#每次训练进行 * 个梯度更新
    #"tau": 0.007, # 【新增】加速目标网络追踪速度 (原默认 0.005),过大导致梯度爆炸
    "gamma": 0.99,#折扣因子，越小越重视短期奖励，最大为 1

    # MAE 梯度控制 - 关键优化
    "dropout": 0.05/scale_ratio,#与 policy_transformer 共用
    "actor_alpha": 0.0,# MAE 反向梯度更新的权重（Actor 端，默认值 0.1）
    "critic_alpha": 0.0,# MAE 反向梯度更新的权重（Critic 端，默认值 1.0）
    "ac_input_dim": 128//scale_ratio,# actor/critic输入维度(到隐藏层转换为1,减一维),对应policy_transformer 的输出层维度（默认值 128）

    # gSDE 探索增强 - 强强联合
    "use_sde": True,          # 启用状态依赖探索，让探索噪声与状态相关，保持策略一致性
    "sde_sample_freq": -1,    # -1 表示每个 Episode 采样一次噪声矩阵（保持整个交易周期的探索一致性）
}

# ===== 完整的 MAESAC 参数（包含不可调节的架构参数） =====
MAESAC_PARAMS = {
    # --- 可调节的训练超参数 ---
    **MAESAC_TUNABLE_PARAMS,

    # --- 不可调节的架构参数（固化在MAE模型中，加载时不可修改） ---
    "enc_in": ENCODER_INPUT_SIZE,#MAE 编码器的输入维度#股票数 88+ 技术指标数 8
    "dec_in": ENCODER_INPUT_SIZE,#MAE 解码器的输入维度
    "c_out_construction": ENCODER_INPUT_SIZE,#MAE 模型的输出维度（只用来评估重建损失）
    "d_ff":256,#demension of Feed-Forward Network(FFN，前馈神经网络),位于 MAE 编码/解码 block 内
    "n_heads":4,#多头注意力机制的头数
    "e_layers":2,#编码器层数
    "d_layers":1,#解码器层数
    "hidden_out":128,#即 d_model, MAE/short/long 模型的隐藏层输出维度（编码后解码前，输入给 SAC 模型 policy_transformer）

    "stock_dim": TICKET_SIZE,
    "transformer_path": '',#mae_model_path,
    "transformer_device": device,

    # 设备配置
    # "optimize_memory_usage": True, # 开启内存优化，减少 ReplayBuffer 占用
    # "replay_buffer_kwargs": {
    #     "handle_timeout_termination": False, # 与 optimize_memory_usage=True 互斥
    # },
}

#策略网络参数 (MlpPolicy Policy Network，包括 act/critic/critic_target)
policy_kwargs = {
    "optimizer_kwargs": {"weight_decay": 1e-2},# 作用：惩罚大的权重值，促使网络权重保持较小，提高泛化能力
    "optimizer_class": torch.optim.AdamW, # 配合weight_decay使用
    "net_arch": [128,128], # 隐藏层维度,默认 [256,256],此处ac_input_dim已转换为1维(减一维)
    "use_sde": True # 保持与 MAESAC_TUNABLE_PARAMS 一致
}

# MAESAC_PARAMS_PRED = {
#     "batch_size": 128,
#     "buffer_size": 50000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1",
#     "enc_in":TEMPORAL_FEATURE_SIZE,#时序特征数 10
#     "dec_in":TEMPORAL_FEATURE_SIZE,
#     "c_out_prediction":1,#不同于 MAESAC_PARAMS
#     "hidden_out":128,#d_model
#     "d_ff":256,
#     "n_heads":8,#不同于 MAESAC_PARAMS
#     "e_layers":3,#不同于 MAESAC_PARAMS
#     "d_layers":2,#不同于 MAESAC_PARAMS
#     "dropout":0.05,
#     "pred_len":1,#不同于 MAESAC_PARAMS
#     "seq_len":60,#不同于 MAESAC_PARAMS
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
    "d_model": 128,                               # dimension of hidden out 
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
    "train_epochs": 15,
    "itr": 1,
    "batch_size": 32,
    "seq_len": 60,
    "label_len": 1,
    "pred_len": 1,
    "enc_in": TEMPORAL_FEATURE_SIZE,#时序特征 10 个
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
    "train_epochs": 15,
    "itr": 1,
    "batch_size": 32,
    "seq_len": 60,
    "label_len": 1,
    "pred_len": 1,
    "enc_in": TEMPORAL_FEATURE_SIZE,#时序特征 10 个
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
    "enc_in": ENCODER_INPUT_SIZE,#编码器输入维度（股票数 88+ 技术指标数 8）
    "dec_in": ENCODER_INPUT_SIZE,#解码器输入维度（股票数 88+ 技术指标数 8）
    "c_out": ENCODER_INPUT_SIZE,#输出维度（股票数 88+ 技术指标数 8）
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
