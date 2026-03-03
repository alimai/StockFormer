# DRL models from Stable Baselines 3

import time

import numpy as np
import pandas as pd
from utils import config
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results




MODELS = {"maesac": SAC_MAE}
#此处导入config中的MAESAC_PARAMS作为默认值备用(注意有参数"c_out_prediction":1)
MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class CheckCallback(BaseCallback):
    def __init__(self, check_freq:int, verbose: int=1):
        super(CheckCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(self.n_calls)

class FinancialMetricsCallback(BaseCallback):
    """
    自定义 Callback：在每个 episode 结束时记录金融指标到 TensorBoard
    """
    def __init__(self, verbose: int = 0):
        super(FinancialMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 使用 VecMonitor 提供的 "episode" 信息作为 episode 结束的唯一信号
        infos = self.locals.get("infos")
        if infos is not None and len(infos) > 0 and "episode" in infos[0]:
            info = infos[0]
            # 记录到 TensorBoard
            if "reward_ratio" in info:
                self.logger.record("finance/reward_ratio", info["reward_ratio"])
            if "reward_step" in info:
                self.logger.record("finance/reward_step", info["reward_step"])
            if "sharpe" in info:
                self.logger.record("finance/sharpe_ratio", info["sharpe"])
        return True

class CombinedCallback(BaseCallback):
    """
    组合回调：包含Tensorboard、模型保存和训练奖励记录功能
    """
    def __init__(self, model_save_path="", check_freq=0, log_dir="", verbose=0):
        super(CombinedCallback, self).__init__(verbose)
        self.model_save_path = model_save_path
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        self.episode_rewards = [] # 用于在内存中存储最近的奖励，避免读取磁盘
        
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 检查episode是否结束，如果是则增加计数并考虑保存模型
        # 使用 VecMonitor 提供的 "episode" 信息作为 episode 结束的唯一信号
        # 这比检查 done/dones 更可靠，能从根本上避免重复计数问题
        infos = self.locals.get("infos")
        if infos is not None and len(infos) > 0 and "episode" in infos[0]:
            self.episode_count += 1
            # 记录奖励到内存列表
            ep_rew = infos[0]["episode"]["r"]
            self.episode_rewards.append(ep_rew)
            if len(self.episode_rewards) > 100: # 只保留最近100个，防止列表无限增长
                self.episode_rewards.pop(0)

            # 每10个episode保存一个备份
            if self.episode_count % 10 == 0:
                tmp_path = os.path.join(self.model_save_path, "tmp_model.zip")
                self.model.save(tmp_path)
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}: Saved checkpoint to {tmp_path}")

        # 按指定频率记录训练奖励
        if self.check_freq > 0 and self.n_calls % self.check_freq == 0:
            # 从内存中获取最近10个episode的平均奖励，不再从磁盘读取
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                if self.verbose > 0:
                    print(f"Best training mean reward: {self.best_mean_reward:.2f} - new mean reward: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best training model to {self.model_save_path}")
                    self.model.save(self.model_save_path+'/best_train_model.zip')

        return True

class FinancialEvalCallback(EvalCallback):
    """
    自定义 EvalCallback：在评估阶段记录金融指标到 TensorBoard
    """
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq, n_eval_episodes, deterministic, render):
        super(FinancialEvalCallback, self).__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render
        )

    def _on_step(self) -> bool:
        # 检查是否到了评估频率
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 为了获取评估期间的金融指标，我们需要运行评估并收集相关信息
            # 由于evaluate_policy不直接返回info字典，我们需要手动运行评估过程
            try:
                # 重置评估环境
                episode_rewards = []
                episode_lengths = []
                
                # 存储每次评估的信息
                eval_info_list = []
                
                for episode in range(self.n_eval_episodes):
                    episode_reward = 0.0
                    episode_length = 0
                    state = self.eval_env.reset()
                    done = [False]
                    
                    while not done[0]:
                        # 预测动作
                        action, _ = self.model.predict(state, deterministic=self.deterministic)                        
                        # 执行动作
                        state, reward, done, info = self.eval_env.step(action)                        
                        # 累积奖励
                        episode_reward += reward[0]
                        episode_length += 1
                        
                        # 如果episode结束，保存最后一步的info
                        if done[0]:
                            eval_info_list.append(info[0])  # info是一个列表，取第一个元素
                            
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                
                # 计算并记录原有指标（eval/mean_reward, eval/mean_ep_length等）
                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                self.last_mean_reward = mean_reward

                if self.verbose > 0:
                    print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                
                # 计算并记录新增的金融指标
                if eval_info_list:
                    reward_ratios = [info.get('reward_ratio', 0) for info in eval_info_list if 'reward_ratio' in info]
                    sharpe_ratios = [info.get('sharpe', 0) for info in eval_info_list if 'sharpe' in info]
                    
                    if reward_ratios:
                        avg_reward_ratio = np.mean(reward_ratios)
                        self.logger.record("eval/reward_ratio", avg_reward_ratio)
                        
                    if sharpe_ratios:
                        avg_sharpe_ratio = np.mean(sharpe_ratios)
                        self.logger.record("eval/sharpe_ratio", avg_sharpe_ratio)

                # 添加到当前Logger
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)

                # Dump log so the evaluation results are printed with the correct timestep
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)

                # 检查是否为最佳模型
                if self.verbose > 0:
                    print(f"Best eval mean reward: {self.best_mean_reward:.2f} - new mean reward: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print(f"Saving new best eval model to {self.best_model_save_path}")

                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = mean_reward

                # 评估完成，直接返回 True 避免 super()._on_step() 再次执行评估
                return True
                        
            except Exception as e:
                # 如果手动评估失败，记录错误但继续执行
                if self.verbose > 0:
                    print(f"Warning: Could not extract financial metrics: {e}")
                
        # 对于非评估步或评估失败的情况，调用父类逻辑
        continue_training = super()._on_step()
        return continue_training


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",#策略网络(MlpPolicy Policy Network,包括act/critic/critic_target)
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, check_freq, model_dir, train_log_dir, eval_log_dir, eval_env, total_timesteps=5000, verbose=1, deterministic=True):
        eval_callback = FinancialEvalCallback(eval_env, best_model_save_path=model_dir, log_path=eval_log_dir, eval_freq=check_freq,
                                                          n_eval_episodes=1, deterministic=deterministic, render=False)
        combined_callback = CombinedCallback(model_save_path=model_dir, check_freq=check_freq, log_dir=train_log_dir, verbose=verbose)
        finance_callback = FinancialMetricsCallback(verbose=verbose)
        callback = CallbackList([eval_callback, combined_callback, finance_callback])

        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback = callback,
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]#, universal_results[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, test_env, cwd, deterministic=True):

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent: 强制设置 buffer_size=1 避免推理阶段分配巨大的内存块
            model = MODELS[model_name].load(cwd, env=test_env, buffer_size=1)
            print("Successfully load model", cwd)
        except Exception as e:
            print(f"Error loading agent from {cwd}: {e}")
            raise e

        # test on the testing env
        state = test_env.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        initial_amount = test_env.env_method(method_name="get_initial_amount")[0]
        episode_total_assets.append(initial_amount)
        done = False
        final_info = None
        
        while not done:
            # 正确解包 predict 返回的元组 (action, states)
            action, _states = model.predict(state, deterministic=deterministic)
            state, reward, dones, info = test_env.step(action)
            
            # DummyVecEnv returns a list/array of done booleans
            done = dones[0]

            # 通过 env_method 获取原始环境的资产信息
            total_asset = test_env.env_method(method_name="get_end_total_asset")[0]

            episode_total_assets.append(total_asset)
            episode_return = total_asset / initial_amount
            episode_returns.append(episode_return)
            
            # done 时保存 info（此时包含 memory 数据，reset 前获取）
            if done:
                final_info = info[0]

        print("episode_return", episode_return)
        print("Test Finished!")

        # 从 terminal 时的 info 获取数据（避免被 DummyVecEnv 自动 reset 清空）
        account_memory = final_info.get('account_memory')
        actions_memory = final_info.get('actions_memory')

        return episode_total_assets, account_memory, actions_memory
