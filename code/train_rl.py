import this
import os
import sys
import time
import datetime
import setuptools

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils import config
from utils.preprocess import FeatureEngineer, data_split
from MySAC.models.DRLAgent import DRLAgent
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env
from sklearn.preprocessing import StandardScaler
from utils.data.stock_data_handle import Stock_Data

working_path = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到模块搜索路径
#sys.path.insert(0, working_path)

if __name__ == '__main__':
    version = 'CSI/'
    model_name='StockFormer/'
    short_prediction_model_path = working_path + '/Transformer/pretrained/csi/Short/checkpoint.pth'
    long_prediction_model_path =  working_path + '/Transformer/pretrained/csi/Long/checkpoint.pth'
    mae_model_path = working_path + '/Transformer/pretrained/csi/mae/checkpoint.pth'
    full_stock_dir = 'data/CSI/'
    ticker_list = config.use_ticker_dict['CSI']
    prediction_len = [1,5]


    if not os.path.exists(config.TRAINED_MODEL_DIR):
        os.makedirs(config.TRAINED_MODEL_DIR)
    if not os.path.exists(config.TENSORBOARD_LOG_DIR):
        os.makedirs(config.TENSORBOARD_LOG_DIR)
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    # 使用 Stock_Data 统一处理所有数据（包括协方差计算和标准化）
    data_manager = Stock_Data(
        root_path='data/', 
        dataset_name='CSI', 
        full_stock_path='CSI/', 
        size=[60, 1, 1], 
        prediction_len=prediction_len
    )

    train = data_manager.get_split_df('train')
    eval = data_manager.get_split_df('valid')
    test = data_manager.get_split_df('test')

    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    tensorboard_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, 'mysac')

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "transaction_cost_pct": 0,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "temporal_feature_list": config.TEMPORAL_FEATURE,
        "type_list": config.TYPE_FEATURE,
        "action_space": stock_dimension,
        "reward_scaling": 100,
        "figure_path":'results/figures/'+version+model_name,
        "csv_path": 'results/csv/'+version+model_name,
        "mode":'train',
        "time_window_start":config.time_window_start,
        "step_len": config.step_len,
        "temporal_len": 60,
        "hidden_channel":128,
        "model_name":model_name[:-1],
        "short_prediction_model_path": short_prediction_model_path,
        "long_prediction_model_path": long_prediction_model_path,
        "device": config.device,
    }

    # evaluation environment
    model_dir = os.path.join(config.TRAINED_MODEL_DIR, version[:-1], model_name[:-1])
    log_dir = os.path.join(config.RESULTS_DIR, version[:-1], model_name[:-1])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Initial Env...")
    train_mode = True#False#
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

        # 【修复】调整包装顺序：只保留 VecMonitor
        env_train_vm = VecMonitor(env_train, log_dir)
        env_eval_vm = VecMonitor(env_eval, log_dir)

        # 【移除】彻底删除 VecNormalize 逻辑，直接使用 VecMonitor 包装后的环境
        agent = DRLAgent(env = env_train_vm)
        if load_pretrain:
            print(f"load: {final_model_path}...")
            model_sac = SAC_MAE.load(final_model_path, env=env_train_vm, tensorboard_log=tensorboard_log_dir)
        else:
            config.MAESAC_PARAMS["transformer_path"] = mae_model_path
            model_sac = agent.get_model("maesac",model_kwargs = config.MAESAC_PARAMS,tensorboard_log=tensorboard_log_dir, seed=config.fix_seed)

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        tb_log_name_with_timestamp = model_name[:-1] + '_' + timestamp + '/'

        print('Start training...')
        start = time.time()
        trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name=tb_log_name_with_timestamp,
                                    check_freq=10000,
                                    log_dir=log_dir,
                                    model_dir=model_dir,
                                    eval_env=env_eval_vm,  # 使用 VecMonitor 包装后的评估环境
                                    total_timesteps=30000) # 提升训练步数至 200,000
        end = time.time()
        print("Training time: %.3f"%(end-start))

    #强化学习训练后保存的模型（如best_train_model.zip）是一个复合模型，它包含了：
    #   - 更新后的MAE模型（state_transformer）---对应原mae/checkpoint.pth
    #   - SAC策略actor网络和价值critic网络 ---全连接层
    #   - 其他Transformer组件（actor_transformer, critic_transformer）
    model_path = os.path.join('trained_models/', version, model_name, 'best_train_model.zip')

    env_name = "test"
    env_kwargs["mode"] = env_name
    env_kwargs["time_window_start"] = [env_kwargs["temporal_len"]]#60
    test_trade_gym = Env(df = test, **env_kwargs)
    env_test, _ = test_trade_gym.get_sb_env()
    # 测试阶段：使用原始环境
    start = time.time()
    results = DRLAgent.DRL_prediction_load_from_file(model_name='maesac',test_env=env_test, cwd=model_path)
    end = time.time()
    print("Test time: %.3f"%(end-start))

    df_root = 'results/test/'+version+model_name
    os.makedirs(df_root, exist_ok=True)
    assets_test, actions_test = results[1], results[2]
    actions_test.to_csv(df_root+'df_actions_test.csv')
    assets_test.to_csv(df_root+'df_assets_test.csv')
