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
        "additional_list": config.ADDITIONAL_FEATURE,
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

        # 【修复】调整包装顺序：先 VecMonitor 再 VecNormalize
        # 这样 VecMonitor 记录的是原始奖励，ep_rew_mean 才能正确反映训练效果
        env_train_vm = VecMonitor(env_train, log_dir+'_train')  # 先包装 Monitor
        env_eval_vm = VecMonitor(env_eval, log_dir+'_test')

        # 使用 VecNormalize 对 reward 进行标准化，避免终端 reward 和普通 reward 数值差异过大
        vn_path = os.path.join('trained_models/', version, model_name, 'vec_normalize.pkl')
        if load_pretrain:#
            if os.path.exists(vn_path):# 恢复 VecNormalize 的统计信息
                env_train_vn = VecNormalize.load(vn_path, env_train_vm)  # 注意改为包装 vm
                env_eval_vn = VecNormalize.load(vn_path, env_eval_vm)
                print(f"Loaded VecNormalize stats from {vn_path}")
            else:
                print("Can not loaded VecNormalize stats!!!")
                exit(0)
        else:
            env_train_vn = VecNormalize(env_train_vm, norm_reward=True, norm_obs=True, gamma=config.MAESAC_PARAMS.get("gamma", 0.99))  # 传入一致的 gamma
            env_eval_vn = VecNormalize(env_eval_vm, norm_reward=True, norm_obs=True, gamma=config.MAESAC_PARAMS.get("gamma", 0.99))

        # 评估环境冻结统计信息，避免评估时更新均值/方差
        env_eval_vn.training = False
        env_eval_vn.norm_reward = False


        # 【修复】使用 VecNormalize 包装后的环境（最外层）
        agent = DRLAgent(env = env_train_vn)
        if load_pretrain:
            print(f"load: {final_model_path}...")
            model_sac = SAC_MAE.load(final_model_path, env=env_train_vn, tensorboard_log=tensorboard_log_dir)
        else:
            config.MAESAC_PARAMS["transformer_path"] = mae_model_path
            model_sac = agent.get_model("maesac",model_kwargs = config.MAESAC_PARAMS,tensorboard_log=tensorboard_log_dir, seed=config.fix_seed)

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        tb_log_name_with_timestamp = model_name[:-1] + '_' + timestamp + '/'

        print('Start training...')
        start = time.time()
        trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name=tb_log_name_with_timestamp,
                                    check_freq=50000,
                                    log_dir=log_dir,
                                    model_dir=model_dir,
                                    eval_env=env_eval_vn,  # 同样使用 VecNormalize 包装后的评估环境
                                    total_timesteps=30000)
        end = time.time()
        print("Training time: %.3f"%(end-start))

        # 保存 VecNormalize 的统计信息
        env_train_vn.save(os.path.join(model_dir, 'vec_normalize.pkl'))

    #强化学习训练后保存的模型（如best_train_model.zip）是一个复合模型，它包含了：
    #   - 更新后的MAE模型（state_transformer）---对应原mae/checkpoint.pth
    #   - SAC策略actor网络和价值critic网络 ---全连接层
    #   - 其他Transformer组件（actor_transformer, critic_transformer）
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
        env_test_vn = VecNormalize(env_test, norm_reward=True, norm_obs=True, gamma=config.MAESAC_PARAMS.get("gamma", 0.99))

    # 测试时冻结 VecNormalize 统计信息，避免测试数据污染训练时的统计
    env_test_vn.training = False  # 停止更新均值/方差统计
    env_test_vn.norm_reward = False  # 测试时不需要归一化奖励

    start = time.time()
    results = DRLAgent.DRL_prediction_load_from_file(model_name='maesac',test_env=env_test_vn, cwd=model_path)
    end = time.time()
    print("Test time: %.3f"%(end-start))

    df_root = 'results/test/'+version+model_name
    os.makedirs(df_root, exist_ok=True)
    assets_test, actions_test = results[1], results[2]
    actions_test.to_csv(df_root+'df_actions_test.csv')
    assets_test.to_csv(df_root+'df_assets_test.csv')
