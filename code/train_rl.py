#import this
import os
import sys
import time
import datetime
import json
import random

from utils import config
from MySAC.Agent.DRLAgent import DRLAgent
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
from stable_baselines3.common.vec_env import VecMonitor
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env
from utils.data.stock_data_handle import Stock_Data
from torch.optim import AdamW

if __name__ == '__main__':
    print("start:")
    fix_seed = random.randint(1, 5000) # 2022 #
    config.set_seed(fix_seed)
    version_name = config.version_name
    model_name = config.model_name

    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    tb_log_name_with_timestamp = model_name + '_' + timestamp + '/'
    print(f"log name: {tb_log_name_with_timestamp}")

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
        prediction_len=prediction_len,
        a=config.SCALE_A
    )

    train = data_manager.get_split_df('train')
    eval = data_manager.get_split_df('valid')
    test = data_manager.get_split_df('test')

    train_data = data_manager.get_split_data('train')
    eval_data = data_manager.get_split_data('valid')
    test_data = data_manager.get_split_data('test')

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
        "mode":'train',
        "model_name":model_name,#'StockFormer'
        "version_name": version_name,#'CSI_2',
        "ratio_max": 0.1,#单股占比
        "initial_amount": 100000,
        "transaction_cost_pct": 0,
        "reward_scaling": 100,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "temporal_feature_list": config.TEMPORAL_FEATURE,
        "type_list": config.TYPE_FEATURE,
        "time_window_start":[i for i in range(60, train_length - config.stride, config.stride)],
        "step_len": config.step_len,
        "temporal_len": 60,
        "hidden_out":config.MAESAC_PARAMS["hidden_out"],#128,MAE/short/long 模型的隐藏层输出维度
        "result_path":config.RESULTS_DIR,#os.path.join(config.RESULTS_DIR, 'figures', version_name, model_name),
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

    final_model_name = 'final_train_model'
    buffer_name = 'replay_buffer'

    train_mode = True#False#
    if train_mode:
        print("Initial train Env...")
        env_kwargs["mode"] = "train"
        train_trade_gym = Env(df = train, data_all = train_data, **env_kwargs)
        env_train, _ = train_trade_gym.get_sb_env()

        print("Initial eval Env...")
        env_kwargs["mode"] = "eval"
        env_kwargs["time_window_start"] = [env_kwargs["temporal_len"]]#60
        eval_trade_gym = Env(df = eval, data_all = eval_data, **env_kwargs)
        env_eval, _ = eval_trade_gym.get_sb_env()

        # 使用 VecMonitor 包装环境以记录训练和评估的统计信息
        env_train_vm = VecMonitor(env_train, log_path_train)
        env_eval_vm = VecMonitor(env_eval, log_path_eval)

        
        if config.struct_base_flag:
            load_model_path = ''
            buffer_path = ''
        else:# 已训练的模型和buffer，如果存在则加载继续训练
            load_model_path = os.path.join(model_path, final_model_name+'_out.zip')
            buffer_path = os.path.join(model_path, buffer_name+'_out.npz')
        
        # 训练强化学习代理，加载模型
        agent = DRLAgent(env = env_train_vm)
        if os.path.exists(load_model_path):
            try:
                print(f"load: {load_model_path}...")
                model_sac = SAC_MAE.load(
                    load_model_path,
                    env=env_train_vm,
                    tensorboard_log=tensorboard_log_dir,
                    policy_kwargs=config.policy_kwargs,
                    **config.MAESAC_TUNABLE_PARAMS#解包传入可调节的超参数
                )
            except Exception as e:
                print(f"Failed to load model: {e}")
                sys.exit(1)#退出训练
        else:
            config.MAESAC_PARAMS["transformer_path"] = mae_model_path
            model_sac = agent.get_model("maesac",model_kwargs = config.MAESAC_PARAMS,
                                        tensorboard_log=tensorboard_log_dir,
                                        seed=fix_seed, 
                                        policy_kwargs=config.policy_kwargs
                                        )

        # 加载已有的 Buffer
        if os.path.exists(buffer_path):
            try:
                max_load = config.MAESAC_PARAMS.get("buffer_max_load")
                print(f"正在加载 Buffer 文件以实现热启动 (max_load={max_load}): {buffer_path}")
                model_sac.load_replay_buffer(buffer_path, max_load=max_load)
                print("Buffer 加载成功！")
            except Exception as e:
                print(f"加载 Buffer 失败，将跳过加载阶段：{e}")

        # 在训练正式开始前保存参数配置到 tensorboard 日志目录
        config_save_dir = os.path.join(tensorboard_log_dir, tb_log_name_with_timestamp)
        os.makedirs(config_save_dir, exist_ok=True)
        with open(os.path.join(config_save_dir, 'maesac_config.json'), 'w', encoding='utf-8') as f:
            # 处理不可序列化项为字符串
            serializable_config = {k: str(v) for k, v in config.MAESAC_TUNABLE_PARAMS.items()}
            serializable_config['struct_base_flag'] = str(config.struct_base_flag)
            serializable_config['fix_seed'] = str(fix_seed)
            json.dump(serializable_config, f, indent=4, ensure_ascii=False)

        print('Start training...')
        start = time.time()
        trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name=tb_log_name_with_timestamp,
                                    check_freq=2000,
                                    train_log_dir=log_path_train,#callback 路径
                                    eval_log_dir=log_path_eval,#callback 路径
                                    model_dir=model_path,
                                    eval_env=env_eval_vm,
                                    total_timesteps=99000)
        end = time.time()
        print("Training time: %.3f"%(end-start))

        # 训练完成后保存最终模型和 buffer
        # 强化学习训练后保存的模型（如 best_train_model.zip）是一个复合模型，它包含了：
        #   - 更新后的 MAE 模型（state_transformer）---对应原 mae/checkpoint.pth
        #   - SAC 策略 actor 网络和价值 critic 网络 ---全连接层
        #   - 其他 Transformer 组件（actor_transformer, critic_transformer）
        final_model_path = os.path.join(model_path, final_model_name+'_out.zip')
        trained_sac.save(final_model_path)
        print(f"最终训练模型已保存到：{final_model_path}")
        buffer_path_out = os.path.join(model_path, buffer_name+'_out.npz')
        trained_sac.save_replay_buffer(buffer_path_out)


    print("Initial test Env...")
    env_kwargs["mode"] = "test"
    env_kwargs["time_window_start"] = [env_kwargs["temporal_len"]]#60
    test_trade_gym = Env(df = test, data_all = test_data, **env_kwargs)
    env_test, _ = test_trade_gym.get_sb_env()
    # 测试阶段：使用原始环境
    test_model_path = os.path.join(model_path, final_model_name+'_out.zip')#'best_train_model.zip')
    results = DRLAgent.DRL_prediction_load_from_file(model_name='maesac',test_env=env_test, cwd=test_model_path)

    df_root = os.path.join(config.RESULTS_DIR, 'test', version_name, model_name)
    os.makedirs(df_root, exist_ok=True)
    assets_test, actions_test = results[1], results[2]
    actions_test.to_csv(os.path.join(df_root, 'df_actions_test.csv'))
    assets_test.to_csv(os.path.join(df_root, 'df_assets_test.csv'))
    print("=================================")
    print("end.")
    print("=================================")
    print("=================================\n\n\n")
