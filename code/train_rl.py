
import os
import time
import datetime
import torch

import pandas as pd
import numpy as np
import matplotlib
import pickle as pkl
matplotlib.use('Agg')
import datetime

from MySAC import config
from MySAC.preprocessors import FeatureEngineer, data_split
from MySAC.models.DRLAgent import DRLAgent
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env
import pdb
import stable_baselines3.common.utils as utils 
from sklearn.preprocessing import StandardScaler


fix_seed = 1999
version = 'CSI/'
model_name='StockFormer/'
short_prediction_model_path = 'Transformer/pretrained/csi/Short/checkpoint.pth' 
long_prediction_model_path =  'Transformer/pretrained/csi/Long/checkpoint.pth'
mae_model_path = 'Transformer/pretrained/csi/mae/checkpoint.pth' 
full_stock_dir = '../data/CSI/'
ticker_list = config.use_ticker_dict['CSI']
prediction_len = [1,5]


if not os.path.exists(config.TRAINED_MODEL_DIR):
    os.makedirs(config.TRAINED_MODEL_DIR)
if not os.path.exists(config.TENSORBOARD_LOG_DIR):
    os.makedirs(config.TENSORBOARD_LOG_DIR)
if not os.path.exists(config.RESULTS_DIR):
    os.makedirs(config.RESULTS_DIR)


df = pd.DataFrame([], columns=['date','open','close','high','low','volume','dopen','dclose','dhigh','dlow','dvolume','price','tic'])

for ticker in ticker_list:
    temp_df = pd.read_csv(os.path.join(full_stock_dir,ticker+'.csv'), usecols=['date', 'open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price'])
    temp_df['date'] = temp_df['date'].apply(lambda x:str(x))
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_df['label_short_term'] = temp_df['close'].pct_change(periods=prediction_len[0]).shift(periods=(-1*prediction_len[0]))
    temp_df['label_long_term'] = temp_df['close'].pct_change(periods=prediction_len[1]).shift(periods=(-1*prediction_len[1]))
    temp_df['tic'] = pd.Series([ticker]*len(temp_df))
    # temp_df = temp_df.rename(columns={'Date':'date', 'Open':'open', 'Close':'close', 'High':'high', 'Low':'low', 'Volume':'volume'})
    df = pd.concat((df, temp_df))

df = df.sort_values(by=['date','tic'])
    
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)

print("generate technical indicator...")
df = fe.preprocess_data(df)

# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
    data_lookback = df.loc[i-lookback:i,:]
    price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close') 
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)
    
    covs = return_lookback.cov().values 
    cov_list.append(covs)


df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)
         

scaler = StandardScaler()
df_data = df[config.TECHNICAL_INDICATORS_LIST]
df_data = df_data.replace([np.inf], config.INF)
df_data = df_data.replace([-np.inf], config.INF*(-1))
data = scaler.fit_transform(df_data.values)
df[config.TECHNICAL_INDICATORS_LIST] = data

train = data_split(df, '2011-01-17','2018-12-28')
eval = data_split(df, '2019-01-02', '2021-12-31')
test = data_split(df,'2018-10-09', '2022-04-16')

stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

tensorboard_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, 'mysac')

# 检测GPU可用性并决定使用GPU还是CPU
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

env_kwargs = {
    "hmax": 100, 
    "initial_amount": 100000,  
    "transaction_cost_pct": 0,
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    "temporal_feature_list": config.TEMPORAL_FEATURE,
    "additional_list": config.ADDITIONAL_FEATURE,
    "action_space": stock_dimension,
    "reward_scaling": 100,
    "figure_path":'results/figures/'+version+model_name,
    "csv_path": 'results/csv/'+version+model_name,
    "mode":'train',
    "time_window_start":config.time_window_start,
    "step_len": 500,
    "temporal_len": 60,
    "hidden_channel":128,     
    "model_name":model_name[:-1],
    "short_prediction_model_path": short_prediction_model_path,
    "long_prediction_model_path": long_prediction_model_path,
    "device": device,
}

# evaluation environment
model_dir = os.path.join(config.TRAINED_MODEL_DIR, version[:-1], model_name[:-1])
log_dir = os.path.join(config.RESULTS_DIR, version[:-1], model_name[:-1])
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print("Initial Env...")
train_mode = False#True
if train_mode:
    env_name = "train"
    env_kwargs["mode"] = env_name
    train_trade_gym = Env(df = train, **env_kwargs)
    env_train, _ = train_trade_gym.get_sb_env()

    env_name = "eval"
    env_kwargs["mode"] = env_name
    env_kwargs["time_window_start"] = [env_kwargs["temporal_len"]]#60
    eval_trade_gym = Env(df = eval, **env_kwargs)
    env_eval, _ = eval_trade_gym.get_sb_env()

    # 检查是否存在已训练的模型，如果存在则加载继续训练
    load_pretrain = False
    final_model_path = os.path.join('trained_models/', version, model_name, 'best_train_model000.zip')
    if os.path.exists(final_model_path):
        load_pretrain = True
        
    # 使用 VecNormalize 对 reward 进行标准化，避免终端 reward 和普通 reward 数值差异过大
    vn_path = os.path.join('trained_models/', version, model_name, 'vec_normalize.pkl')
    if load_pretrain:#
        if os.path.exists(vn_path):# 恢复 VecNormalize 的统计信息
            env_train_vn = VecNormalize.load(vn_path, env_train)
            env_eval_vn = VecNormalize.load(vn_path, env_eval)
            print(f"Loaded VecNormalize stats from {vn_path}")
        else:
            print(f"Can not loaded VecNormalize stats!!!")
    else:
        env_train_vn = VecNormalize(env_train, norm_reward=True, norm_obs=True)
        env_eval_vn = VecNormalize(env_eval, norm_reward=True, norm_obs=True)
    env_train_vm = VecMonitor(env_train_vn, log_dir+'_train')
    env_eval_vm = VecMonitor(env_eval_vn, log_dir+'_test')

    MAESAC_PARAMS = {
        "batch_size": 32,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
        "enc_in": 96,
        "dec_in": 96,
        "c_out_construction": 96,
        "d_model":128,
        "d_ff":256,
        "n_heads":4,
        "e_layers":2,
        "d_layers":1,
        "dropout":0.05,
        "transformer_path":mae_model_path,
        "transformer_device": device,
    }

    agent = DRLAgent(env = env_train_vm)
    if load_pretrain:
        print(f"load: {final_model_path}...")
        model_sac = SAC_MAE.load(final_model_path, env=env_train_vm, tensorboard_log=tensorboard_log_dir)
    else:
        model_sac = agent.get_model("maesac",model_kwargs = MAESAC_PARAMS,tensorboard_log=tensorboard_log_dir, seed=fix_seed)

    timestamp = datetime.datetime.now().strftime("%H%M%S")
    tb_log_name_with_timestamp = model_name[:-1] + '_' + timestamp + '/'

    print('Start training...')
    start = time.time()
    trained_sac = agent.train_model(model=model_sac,
                                tb_log_name=tb_log_name_with_timestamp,
                                check_freq=50000,
                                log_dir=log_dir,
                                model_dir=model_dir,
                                eval_env=env_eval_vm,
                                total_timesteps=30000)
    end = time.time()
    print("Training time: %.3f"%(end-start))

    # 保存 VecNormalize 的统计信息
    env_train_vn.save(os.path.join(model_dir, 'vec_normalize.pkl'))


model_path = os.path.join('trained_models/', version, model_name, 'best_train_model.zip')
vn_path = os.path.join('trained_models/', version, model_name, 'vec_normalize.pkl')

env_name = "test"
env_kwargs["mode"] = env_name
env_kwargs["time_window_start"] = [env_kwargs["temporal_len"]]#60
test_trade_gym = Env(df = test, **env_kwargs)
env_test, _ = test_trade_gym.get_sb_env()
# 如果存在 VecNormalize 统计文件，则加载它
if os.path.exists(vn_path):
    env_test_vn = VecNormalize.load(vn_path, env_test)
    print(f"Loaded VecNormalize from {vn_path}")
else:
    env_test_vn = VecNormalize(env_test, norm_reward=True, norm_obs=True)
#env_test_vm = VecMonitor(env_test_vn, log_dir+'_test')

start = time.time()
results = DRLAgent.DRL_prediction_load_from_file(model_name='maesac',test_env=env_test_vn, cwd=model_path)
end = time.time()
print("Test time: %.3f"%(end-start))

df_root = 'results/df_print/'+version+model_name
os.makedirs(df_root, exist_ok=True)
assets_his, df_actions = results[1], results[2]
df_actions.to_csv(df_root+'df_actions_test.csv')
assets_his.to_csv(df_root+'df_assets_his_test.csv')



