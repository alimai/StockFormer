import numpy as np
import pandas as pd
import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from Transformer.models.transformer import Transformer_base as PredictionModel

import torch
from collections import OrderedDict

import os


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gymnasium"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,#state_space = stock_num
        action_space,
        tech_indicator_list,
        temporal_feature_list,
        type_list,
        time_window_start, # should be a list
        short_prediction_model_path = None,
        long_prediction_model_path = None,
        step_len=1000,
        temporal_len=60,
        figure_path='results/',
        csv_path = 'results/',
        mode="train",
        hidden_channel=128,
        make_plots=True,
        print_verbosity=1,
        initial=True,
        model_name="",
        iteration="",
        device=None,
        print_additional_flag=0,
        data_all=None, # 新增参数
    ):
        super().__init__()
        # start time
        self.start_day = time_window_start[0]
        self.day = self.start_day
        self.time_window_start = time_window_start
        self.time_windows_point = 0
        self.step_len = step_len

        # help file
        self.figure_path = figure_path
        self.csv_path = csv_path
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)

        self.df = df
        self.data_all = data_all # 保存全量数据矩阵
        self.stock_dim = stock_dim
        
        # 优化：预先提取价格和日期，避免 step 中的 Pandas 索引
        self.prices_all = self.df['price'].values.reshape(-1, self.stock_dim).astype(np.float32)
        self.dates_all = self.df['date'].unique()
        self.max_day = len(self.dates_all) - 1
        
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.transaction_cost_pct = transaction_cost_pct

        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_dim = action_space
        self.tech_indicator_list = tech_indicator_list
        self.temporal_feature_list = temporal_feature_list
        self.type_list = type_list
        self.temporal_len = temporal_len
        self.hidden_channel = hidden_channel

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        tech_dim = len(self.tech_indicator_list)#8
        # cov matrix list + technical list + temporal feature * 60 + prediction labels + month_day (7) + weekday (5)
        # observation_space：self._update_state()生成数据维度；Modified: date features now take 12 dimensions (One-hot)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self.state_space+tech_dim+self.hidden_channel*2+12))
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self.state_space + tech_dim  + 12))
        
        # Modified: Update hidden_state_space to strictly include MAE output (128)# + Tech + Date
        # This excludes redundant Covariance data from the SAC input stream
        # hidden_state_space: actor/critic输入维度，对应 actor_transformer/critic_transformer输出维度
        # 亦即policy_transformer_stock_atten2.forward()生成数据维度
        self.hidden_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self.hidden_channel*3 + tech_dim + 12))
        
        # observation_space用于指定state的维度，hidden_state_space用于指定SAC的输入维度
        # 二者在最后的+m/+n的不同表示输出了m维额外信息，但SAC只接受n维额外信息(差值在policy_transformer_stock_atten2中处理)
        print("action_space shape: ",self.action_space.shape)
        print("observation_space shape: ",self.observation_space.shape)
        print("hidden_state_space shape: ",self.hidden_state_space.shape)

        self.data = self.data_all[self.day] # 使用 numpy 索引
        self.tic = self.df.tic.unique()
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # load model - 检测GPU可用性并决定使用GPU还是CPU
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        self.device = device
        self.short_prediction_model = self.load_model(short_prediction_model_path).to(self.device)
        self.long_prediction_model = self.load_model(long_prediction_model_path).to(self.device)
        self.short_prediction_model.eval()
        self.long_prediction_model.eval()
        
        # 预计算所有时间步的 hidden state
        self._precompute_hidden_states()

        self.warmup_steps = 15 # 新增：预热步数

        # additional list
        self.print_additional_flag = print_additional_flag
        self.short_hidden_feature = []
        self.long_hidden_feature = []

        # 优化：env_info 初始化为 Numpy 数组 [cash, prices..., shares...]
        self.env_info = np.zeros(1 + 2 * self.stock_dim, dtype=np.float32)
        self.env_info = self._initiate_info()
        self.state = self._initial_state()

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.amount_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        # self.reset()
        self._seed()


    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.env_info[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.env_info[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.env_info[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.env_info[index + 1]
                        * sell_num_shares
                        * (1 - self.transaction_cost_pct)
                    )
                    # update balance
                    self.env_info[0] += sell_amount

                    self.env_info[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.env_info[index + 1] * sell_num_shares * self.transaction_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.env_info[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.env_info[0] // self.env_info[index + 1]
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.env_info[index + 1] * buy_num_shares * (1 + self.transaction_cost_pct)
                )
                self.env_info[0] -= buy_amount

                self.env_info[index + self.stock_dim + 1] += buy_num_shares

                self.cost += self.env_info[index + 1] * buy_num_shares * self.transaction_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        buy_num_shares = _do_buy()

        return buy_num_shares


    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(self.figure_path+"/account_value_{}_{}.png".format(self.mode, self.episode))
        plt.close()

    def _get_future_price(self, days_ahead=5):
        future_day = min(self.day + days_ahead, self.max_day)
        return self.prices_all[future_day]

    def step(self, actions):
        self.terminal = False
        if self.mode == 'train':
            self.terminal = (self.day - self.start_day) >= self.step_len + 1
        if not self.terminal:
            self.terminal = self.day >= self.max_day

        if self.terminal:
            # 使用 Numpy 快速计算总资产
            prices = self.env_info[1 : 1 + self.stock_dim]
            shares = self.env_info[1 + self.stock_dim : 1 + 2 * self.stock_dim]
            self.end_total_asset = self.env_info[0] + np.sum(prices * shares)
            
            tot_reward = (self.end_total_asset - self.initial_amount)
            tot_reward_ratio = tot_reward / self.initial_amount

            # 计算市场增长基准 (Numpy 快速切片)
            start_prices = self.prices_all[self.start_day]
            curr_prices = self.prices_all[self.day]
            market_value_growth_ratio = np.sum(curr_prices) / np.sum(start_prices) - 1.0

            # Numpy 计算 Sharpe
            assets = np.array(self.asset_memory)
            returns = np.diff(assets) / (assets[:-1] + 1e-8)
            sharpe = 0.0
            if len(returns) > 1 and np.std(returns) != 0:
                sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)

            self.reward = tot_reward_ratio + (tot_reward_ratio - market_value_growth_ratio) * 1.5
            self.reward /= (self.day - self.start_day + 1)
            self.reward *= self.reward_scaling

            if self.episode % self.print_verbosity == 0:
                print(self.mode, f"episode: {self.episode}")
                print(f"startday: {self.start_day}, endday: {self.day}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {self.end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if self.make_plots and self.model_name != "" and self.mode != "":
                self._make_plot()
                
                df_total_value = self.save_asset_memory()
                df_rewards = pd.DataFrame(self.rewards_memory, columns=["account_rewards"])
                
                df_actions = self.save_action_memory()
                df_actions.to_csv(self.csv_path+"/actions_{}_{}.csv".format(self.mode, self.episode))
                df_stock_amount = self.save_holding_amount()
                df_stock_amount.to_csv(self.csv_path+"/amount_{}_{}.csv".format(self.mode, self.episode))
                df_total_value.to_csv(self.csv_path+"/account_value_{}_{}.csv".format(self.mode, self.episode), index=False)
                df_rewards.to_csv(self.csv_path+"/account_rewards_{}_{}.csv".format(self.mode, self.episode), index=False)

            # 在 info 中返回 memory 数据
            return self.state, self.reward, self.terminal, False, {
                'reward_ratio': tot_reward_ratio,
                'reward_step': np.sum(self.rewards_memory) if self.rewards_memory else 0.0,
                'sharpe': sharpe,
                'account_memory': df_total_value if self.make_plots else None,
                'actions_memory': df_actions if self.make_plots else None,
            }

        else:
            # 正常交易逻辑步进
            zero_day_prices = self.env_info[1 : 1 + self.stock_dim]
            first_day_prices = self._get_future_price(days_ahead=1)
            fifth_day_prices = self._get_future_price(days_ahead=5)

            shares = self.env_info[1 + self.stock_dim : 1 + 2 * self.stock_dim]
            begin_total_asset = self.env_info[0] + np.sum(zero_day_prices * shares)

            actions = (actions + 1) * self.hmax / 2
            actions = actions.astype(int)
            actions = actions - shares

            argsort_actions = np.argsort(actions)
            
            sell_num = (actions < 0).sum()
            buy_num = (actions > 0).sum()            
            sell_index = argsort_actions[:sell_num]
            buy_index = argsort_actions[::-1][:buy_num]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            # 更新后计算
            self.end_total_asset = self.env_info[0] + np.sum(first_day_prices * shares)

            avg_prices = first_day_prices * 0.7 + fifth_day_prices * 0.3
            asset_for_reward_new = self.env_info[0] + np.sum(avg_prices * shares)
            
            market_value_growth_ratio = np.sum(avg_prices) / np.sum(zero_day_prices) - 1.0
            self.reward = asset_for_reward_new / begin_total_asset - 1.0
            self.reward = self.reward + (self.reward - market_value_growth_ratio) * 1.5
            self.reward *= self.reward_scaling

            self.actions_memory.append(actions)
            self.asset_memory.append(self.end_total_asset)
            self.date_memory.append(self._get_date())
            self.rewards_memory.append(self.reward)
            self.amount_memory.append(shares.copy())

            self.day += 1
            self.data = self.data_all[self.day]
            self.env_info = self._update_info()
            self.state = self._update_state()

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.terminal:#训练或测试后重新初始化
            self.time_windows_point += 1#remove for test
            if self.time_windows_point >= len(self.time_window_start):
                self.time_windows_point = 0
            self.start_day = self.time_window_start[self.time_windows_point]
            if self.mode == 'train':
                rand_start_bias= random.randint(0, int(self.step_len/2))
                self.start_day += rand_start_bias
            self.episode += 1
            self.terminal = False
        else:#最开始步初始化
            self.time_windows_point = 0
            self.start_day = self.time_window_start[self.time_windows_point]
            self.episode = 1


        self.day = self.start_day
        self.data = self.data_all[self.day] # 使用 numpy 索引
        # self.covs = self.data['cov_list'].values[0]

        # 标准初始化
        self.env_info = self._initiate_info()
        self.state = self._initial_state()
        # 重置特征列表
        self.short_hidden_feature = []
        self.long_hidden_feature = []

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [self.initial_amount]
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.amount_memory = []#[self.env_info[-self.stock_dim:]]
        self.date_memory = [self._get_date()]

        print("=================================")
        print(self.mode, f"reset...")
        print("=================================")

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_info(self):
        # env_info: [cash, prices..., shares...]
        info = np.zeros(1 + 2 * self.stock_dim, dtype=np.float32)
        info[0] = self.initial_amount
        info[1 : 1 + self.stock_dim] = self.prices_all[self.day]
        info[1 + self.stock_dim : ] = 0
        return info

    def _initial_state(self):
        # 提取协方差矩阵 (前 stock_dim 列)
        covs = self.data[:, :self.stock_dim]
        
        # 提取技术指标
        tech_start = self.stock_dim
        tech_end = tech_start + len(self.tech_indicator_list)
        technical_indicators = self.data[:, tech_start:tech_end]


        # 使用预计算的 hidden features
        hidden_np1 = self.precomputed_short[self.day]
        hidden_np2 = self.precomputed_long[self.day]

        self.short_hidden_feature.append(hidden_np1)
        self.long_hidden_feature.append(hidden_np2)

        # 提取日期特征 (最后 12 列)
        date_features = self.data[:, -12:]

        #state = np.concatenate((covs, technical_indicators, date_features), axis=-1)
        state = np.concatenate((covs, technical_indicators, hidden_np1, hidden_np2, date_features), axis=-1)
        return state


    def _update_info(self):
        # 原位更新价格部分，保持 Numpy 数组性质
        self.env_info[1 : 1 + self.stock_dim] = self.prices_all[self.day]
        return self.env_info

    def _update_state(self):
        covs = self.data[:, :self.stock_dim]
        
        tech_start = self.stock_dim
        tech_end = tech_start + len(self.tech_indicator_list)
        technical_indicators = self.data[:, tech_start:tech_end]
        

        # 使用预计算的 hidden features

        hidden_np1 = self.precomputed_short[self.day]

        hidden_np2 = self.precomputed_long[self.day]



        # # 优化：限制hidden feature列表的最大长度，避免内存累积
        max_hidden_length = 30  # 最多保存最近50个时间步的特征
        if len(self.short_hidden_feature) >= max_hidden_length:
            self.short_hidden_feature.pop(0)
            self.long_hidden_feature.pop(0)

        self.short_hidden_feature.append(hidden_np1)
        self.long_hidden_feature.append(hidden_np2)

        # 提取日期特征 (最后 12 列)
        date_features = self.data[:, -12:]

        #state = np.concatenate((covs, technical_indicators, date_features), axis=-1)
        state = np.concatenate((covs, technical_indicators, hidden_np1, hidden_np2, date_features), axis=-1)
        # print("Update: ",state.shape)
        return state

    def _get_date(self):
        # 使用缓存的日期数组，秒回
        return self.dates_all[self.day]

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def get_end_total_asset(self):
        return self.end_total_asset

    def get_initial_amount(self):
        return self.initial_amount

    def save_additional_info(self):
        # temp_dict = {"short_hidden_feature":self.short_hidden_feature, "long_hidden_feature": self.long_hidden_feature}
        temp_dict = {"short_hidden_feature":[], "long_hidden_feature": []}
        return temp_dict

    def save_holding_amount(self):
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        amount_list = self.amount_memory
        df_amount = pd.DataFrame(amount_list)
        df_amount.columns = self.tic
        return df_amount


    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.tic
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def load_model(self, path, enc_in=10, dec_in=10, c_out=1):
        model = PredictionModel(enc_in=enc_in, dec_in=dec_in, c_out=c_out)

        if path is not None:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            
            # 检查是否为DataParallel保存的模型（键名带有"module."前缀）
            if any(k.startswith('module.') for k in state_dict.keys()):
                # 移除"module."前缀
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # 移除"module."前缀
                    new_state_dict[name] = v
            else:
                # 如果没有"module."前缀，则直接使用原始state_dict
                new_state_dict = state_dict
            
            model.load_state_dict(new_state_dict)
            print("Successfully load prediction mode...", path)

        return model

    def _precompute_hidden_states(self):
        print("Precomputing hidden states for all days...")
        
        # Extract all features into a 3D numpy array: (Total_Days, Stock_Dim, Feature_Dim)
        # Assuming self.df is sorted by date then tic (which is the case as per reshape usage in init)
        
        total_rows = len(self.df)
        total_days = len(self.dates_all)
        feature_dim = len(self.temporal_feature_list)
        
        all_features = self.df[self.temporal_feature_list].values
        # Reshape: (Total_Days, Stock_Dim, Feature_Dim)
        # 确保数据按 (Day, Stock, Feature) 排列
        all_features = all_features.reshape(total_days, self.stock_dim, feature_dim)
        
        # 初始化存储数组
        self.precomputed_short = np.zeros((total_days, self.stock_dim, self.hidden_channel), dtype=np.float32)
        self.precomputed_long = np.zeros((total_days, self.stock_dim, self.hidden_channel), dtype=np.float32)
        
        batch_size_days = 64 # 每次处理 64 天的数据 -> 64 * 88 = 5632 样本
        
        start_idx = self.temporal_len - 1
        end_idx = total_days
        
        with torch.no_grad():
            for i in range(start_idx, end_idx, batch_size_days):
                current_batch_days = min(batch_size_days, end_idx - i)
                indices = range(i, i + current_batch_days)
                
                # Prepare batch input
                batch_input = []
                for day_idx in indices:
                    # Window: [day_idx - temporal_len + 1 : day_idx + 1]
                    window = all_features[day_idx - self.temporal_len + 1 : day_idx + 1]
                    # Transpose to (Stock_Dim, Temporal_Len, Feature_Dim)
                    window = window.transpose(1, 0, 2)
                    batch_input.append(window)
                
                # Stack: (Batch_Days, Stock_Dim, T_Len, F_Dim)
                batch_input_np = np.stack(batch_input)
                
                # Flatten to (Batch_Days * Stock_Dim, T_Len, F_Dim)
                flat_input_np = batch_input_np.reshape(-1, self.temporal_len, feature_dim)
                
                enc_feature = torch.FloatTensor(flat_input_np).to(self.device)
                dec_feature = enc_feature[:, -1:, :] # Last time step
                
                _, hidden_short, _ = self.short_prediction_model(enc_feature, dec_feature)
                _, hidden_long, _ = self.long_prediction_model(enc_feature, dec_feature)
                
                # Reshape back and store
                # Output: (Batch_Days * Stock_Dim, Hidden_Dim)
                hidden_short_np = hidden_short.cpu().numpy().reshape(current_batch_days, self.stock_dim, -1)
                hidden_long_np = hidden_long.cpu().numpy().reshape(current_batch_days, self.stock_dim, -1)
                
                # 生成伪hidden feature数据
                # hidden_feature_dim = self.hidden_channel
                # hidden_short_np = np.random.randn(self.stock_dim, hidden_feature_dim).astype(np.float32)
                # hidden_long_np = np.random.randn(self.stock_dim, hidden_feature_dim).astype(np.float32)
        
                self.precomputed_short[i : i + current_batch_days] = hidden_short_np
                self.precomputed_long[i : i + current_batch_days] = hidden_long_np
                
                if i % (batch_size_days * 5) == 0:
                    print(f"Precomputed {i}/{total_days} days...")
                
        print("Precomputation complete.")
