# DRL models from Stable Baselines 3

import time

import numpy as np
import pandas as pd
from utils import config
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
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
        # 安全地检测 episode 是否结束
        # 不同的 SB3 版本或环境包装器可能使用 "done" (bool) 或 "dones" (list/array)
        done = self.locals.get("done", False)
        dones = self.locals.get("dones")
        
        # 如果 dones 是列表且第一个元素为 True，或者单变量 done 为 True
        is_episode_finished = done or (dones is not None and dones[0])

        if is_episode_finished:
            # 获取 infos 列表
            infos = self.locals.get("infos")
            if infos is not None and len(infos) > 0:
                info = infos[0]
                # 记录到 TensorBoard
                if "reward_ratio" in info:
                    self.logger.record("finance/reward_ratio", info["reward_ratio"])
                if "reward_step" in info:
                    self.logger.record("finance/reward_step", info["reward_step"])
                if "sharpe" in info:
                    self.logger.record("finance/sharpe_ratio", info["sharpe"])
        return True

class SaveModelCallback(BaseCallback):
    """
    自定义 Callback：用于定期保存模型
    - 每 2 个 episode 结束保存为 tmp_mode.zip
    """
    def __init__(self, model_save_path: str, verbose: int = 0):
        super(SaveModelCallback, self).__init__(verbose)
        self.model_save_path = model_save_path
        self.episode_count = 0
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        done = self.locals.get("done", False)
        dones = self.locals.get("dones")
        is_episode_finished = done or (dones is not None and dones[0])

        if is_episode_finished:
            self.episode_count += 1
            
            # 每 2 个 episode 保存一个备份
            if self.episode_count % 2 == 0:
                tmp_path = os.path.join(self.model_save_path, "tmp_mode.zip")
                self.model.save(tmp_path)
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}: Saved checkpoint to {tmp_path}")
        return True

class TrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq:int, model_save_path: str, log_dir: str, verbose: int=1):
        super(TrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = model_save_path
        self.best_mean_reward = -np.inf
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 10 episodes
              mean_reward = np.mean(y[-10:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path+'/best_train_model.zip')

        return True      


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, model_save_path=""):
        super(TensorboardCallback, self).__init__(verbose)
        self.save_path = model_save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:        
        return True


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
        policy="MlpPolicy",
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

    def train_model(self, model, tb_log_name, check_freq, model_dir, log_dir, eval_env, total_timesteps=5000, verbose=1, deterministic=True):
        eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir, log_path=log_dir, eval_freq=check_freq, n_eval_episodes=1, deterministic=deterministic, render=False)
        tb_callback=TensorboardCallback(verbose=verbose, model_save_path=model_dir)
        finance_callback = FinancialMetricsCallback(verbose=verbose)
        save_callback = SaveModelCallback(model_save_path=model_dir, verbose=verbose)
        trainingreward_callback = TrainingRewardCallback(check_freq=check_freq, model_save_path=model_dir, log_dir=log_dir, verbose=verbose)
        callback = CallbackList([eval_callback, tb_callback, finance_callback, save_callback, trainingreward_callback])

        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback = callback,
            model_save_path = model_dir,
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
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException:
            raise ValueError("Fail to load agent!")

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
            state, reward, done, info = test_env.step(action)

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
