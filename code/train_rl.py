#import this
import os
import sys
import time
import datetime

from utils import config
from MySAC.models.DRLAgent import DRLAgent
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env
from utils.data.stock_data_handle import Stock_Data


if __name__ == '__main__':
    
    version_name = config.version_name
    model_name = config.model_name

    working_path = os.path.dirname(os.path.abspath(__file__))
    short_prediction_model_path = working_path + '/../trained_models/'+version_name+'/Short/checkpoint.pth'
    long_prediction_model_path =  working_path + '/../trained_models/'+version_name+'/Long/checkpoint.pth'
    mae_model_path = working_path + '/../trained_models/'+version_name+'/mae/checkpoint.pth'


    if not os.path.exists(config.TRAINED_MODEL_DIR):
        os.makedirs(config.TRAINED_MODEL_DIR)
    if not os.path.exists(config.TENSORBOARD_LOG_DIR):
        os.makedirs(config.TENSORBOARD_LOG_DIR)
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    # 使用 Stock_Data 统一处理所有数据（包括协方差计算和标准化）
    full_stock_dir = os.path.join('data', version_name)
    prediction_len = [1,5]
    data_manager = Stock_Data(
        full_stock_path=full_stock_dir, 
        temporal_len=60,
        prediction_len=prediction_len
    )

    train = data_manager.get_split_df('train')
    eval = data_manager.get_split_df('valid')
    test = data_manager.get_split_df('test')
    # 输出数据范围和维度信息，便于调试和验证
    print(f"Train Date Range: {train['date'].min().date()} - {train['date'].max().date()}")
    print(f"Validation Date Range: {eval['date'].min().date()} - {eval['date'].max().date()}")
    print(f"Test Date Range: {test['date'].min().date()} - {test['date'].max().date()}")

    # 各组数据时间步长度（日期长度）
    # 注意：由于数据处理中包含了多只股票的数据，实际长度为交易日数量×股票数量
    train_length = train['date'].nunique()  # 获取唯一日期数量
    eval_length = eval['date'].nunique()    # 获取唯一日期数量
    test_length = test['date'].nunique()    # 获取唯一日期数量    
    print(f"Train length: {train_length}, Eval length: {eval_length}, Test length: {test_length}")

    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension #卷积，二者相等
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

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
        "figure_path":os.path.join(config.RESULTS_DIR, 'figures', version_name, model_name),
        "csv_path": os.path.join(config.RESULTS_DIR, 'csv', version_name, model_name),
        "mode":'train',
        "time_window_start":[i for i in range(60, train_length - config.step_len, config.stride)],
        "step_len": config.step_len,
        "temporal_len": 60,
        "hidden_channel":128,
        "model_name":model_name,
        "short_prediction_model_path": short_prediction_model_path,
        "long_prediction_model_path": long_prediction_model_path,
        "device": config.device,
    }

    tensorboard_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, 'mysac_tb')
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    log_path = os.path.join(config.TENSORBOARD_LOG_DIR, 'mysac_mnt')
    os.makedirs(log_path, exist_ok=True)
    log_path_train = os.path.join(log_path, 'train')
    os.makedirs(log_path_train, exist_ok=True)
    log_path_eval = os.path.join(log_path, 'eval')
    os.makedirs(log_path_eval, exist_ok=True)
    model_path = os.path.join(config.TRAINED_MODEL_DIR, version_name, model_name)
    os.makedirs(model_path, exist_ok=True)

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
        final_model_path = os.path.join(config.TRAINED_MODEL_DIR, version_name, model_name, 'best_train_model---.zip')
        if os.path.exists(final_model_path):
            load_pretrain = True

        # 使用 VecMonitor 包装环境以记录训练和评估的统计信息
        env_train_vm = VecMonitor(env_train, log_path_train)
        env_eval_vm = VecMonitor(env_eval, log_path_eval)

        # 训练强化学习代理,加载模型
        agent = DRLAgent(env = env_train_vm)
        policy_kwargs = {"optimizer_kwargs": {"weight_decay": 1e-4},
                         "net_arch": [128, 128], # 与 d_model 保持一致，默认[256,256]
                         "use_sde": False
                        }#for MlpPolicy
        if load_pretrain:
            print(f"load: {final_model_path}...")
            model_sac = SAC_MAE.load(final_model_path, env=env_train_vm, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
        else:
            config.MAESAC_PARAMS["transformer_path"] = mae_model_path
            model_sac = agent.get_model("maesac",model_kwargs = config.MAESAC_PARAMS,tensorboard_log=tensorboard_log_dir, seed=config.fix_seed, policy_kwargs=policy_kwargs)

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        tb_log_name_with_timestamp = model_name + '_' + timestamp + '/'

        print('Start training...')
        start = time.time()
        trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name=tb_log_name_with_timestamp,
                                    check_freq=2000,
                                    train_log_dir=log_path_train,#callback路径
                                    eval_log_dir=log_path_eval,#callback路径
                                    model_dir=model_path,
                                    eval_env=env_eval_vm,
                                    total_timesteps=30000)
        end = time.time()
        print("Training time: %.3f"%(end-start))

    #强化学习训练后保存的模型（如best_train_model.zip）是一个复合模型，它包含了：
    #   - 更新后的MAE模型（state_transformer）---对应原mae/checkpoint.pth
    #   - SAC策略actor网络和价值critic网络 ---全连接层
    #   - 其他Transformer组件（actor_transformer, critic_transformer）
    model_path = os.path.join(config.TRAINED_MODEL_DIR, version_name, model_name, 'best_model.zip')

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

    df_root = os.path.join(config.RESULTS_DIR, 'test', version_name, model_name)
    os.makedirs(df_root, exist_ok=True)
    assets_test, actions_test = results[1], results[2]
    actions_test.to_csv(os.path.join(df_root, 'df_actions_test.csv'))
    assets_test.to_csv(os.path.join(df_root, 'df_assets_test.csv'))
