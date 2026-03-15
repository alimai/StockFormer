from typing import Any, Dict, List, Optional, Tuple, Type, Union
import random
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from torch.nn import functional as F
from collections import OrderedDict


from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

from Transformer.models.transformer import Transformer_base as Transformer
from utils.metrics import ranking_loss
from MySAC.SAC.policy_transformer import policy_transformer_stock_atten2 as policy_transformer_attn2
#import pdb


from stable_baselines3 import SAC as SAC_SB3

class SAC(SAC_SB3):
    """
    Soft Actor-Critic (SAC)
    ...
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every "train_freq" steps. Alternatively pass a tuple of frequency and unit
        like "(5, "step")" or "(2, "episode")".
    :param gradient_steps: How many gradient steps to do after each rollout (see "train_freq")
        Set to "-1" means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance "HerReplayBuffer").
        If "None", it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every "target_network_update_freq"
        gradient steps.
    :param target_entropy: target entropy when learning "ent_coef" ("ent_coef = 'auto'")
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        enc_in=96,
        dec_in=96,
        c_out_construction=96,
        hidden_out=128,#即 MAE/short/long 模型的隐藏层输出维度
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        dropout=0.05,
        transformer_device = None,
        transformer_path = None,
        actor_alpha=0.1,
        critic_alpha=1.0,
        stock_dim=88,
        ac_input_dim=128,
        **kwargs,
    ):
        # 【关键修复】在 super().__init__ 之前获取并设置隐藏状态空间
        # 否则父类初始化过程中调用 _setup_model 时会因找不到 hidden_state_space 报错
        # hidden_state_space: actor/critic输入维度，对应 actor_transformer/critic_transformer输出维度
        # 亦即policy_transformer_stock_atten2.forward()生成数据维度
        self.hidden_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(stock_dim, ac_input_dim))
        
        super(SAC, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,#使用 _update_learning_rate() 后学习速率真正被赋初值的地方
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            ent_coef=ent_coef,
            _init_setup_model=False,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        if _init_setup_model:
            self._setup_model()

        # 检测GPU可用性并决定使用GPU还是CPU
        if transformer_device is None:
            if th.cuda.is_available():
                transformer_device = 'cuda:0'
            else:
                transformer_device = 'cpu'

        # MAE 模型
        self.state_transformer = Transformer(enc_in=enc_in, dec_in=dec_in, c_out=c_out_construction,
                                             n_heads=n_heads, e_layers=e_layers, d_layers=d_layers,
                                             d_model=hidden_out, d_ff=d_ff, dropout=dropout).to(transformer_device)

        if transformer_path is not None and transformer_path != '':
            state_dict = th.load(transformer_path, map_location=transformer_device, weights_only=True)
            
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
            
            self.state_transformer.load_state_dict(new_state_dict)
            print("Successfully load pretrained MAE model...", transformer_path)
        else:
            print("Successfully initialize MAE model...")
        
        #for MAE
        self.transformer_device = transformer_device
        self.transformer_optim = th.optim.AdamW(self.state_transformer.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.transformer_criteria = th.nn.MSELoss()
        self.env_hidden_dim = hidden_out

        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha

        # 向 Policy Transformer 传递附加信号维度(Tech + Date)
        # 使用环境隐藏维度进行计算，确保与环境生成的 Observation 结构对齐
        additional_dim = env.observation_space.shape[1] - stock_dim  - self.env_hidden_dim * 2
        # additional_dim = env.observation_space.shape[1] - stock_num  - hidden_out* 2
        
        self.actor_transformer = policy_transformer_attn2(hidden_out=hidden_out, dropout=dropout, lr=learning_rate, device=transformer_device, additional_dim=additional_dim).to(transformer_device)
        self.critic_transformer = policy_transformer_attn2(hidden_out=hidden_out, dropout=dropout, lr=learning_rate, device=transformer_device, additional_dim=additional_dim).to(transformer_device)
        self.critic_transformer_target = policy_transformer_attn2(hidden_out=hidden_out, dropout=dropout, lr=learning_rate, device=transformer_device, additional_dim=additional_dim).to(transformer_device)
        self.critic_transformer_target.load_state_dict(self.critic_transformer.state_dict())
        self.critic_transformer_target.eval()

        # 【修复】将内部优化器提升为直接属性，方便 SB3 通过 getattr 加载（SB3 不支持 'a.b' 这种路径）
        self.actor_transformer_optim = self.actor_transformer.optimizer
        self.critic_transformer_optim = self.critic_transformer.optimizer

        # self.in_feat (enc_in) = stock_num + tech_dim = 96
        self.in_feat = enc_in

    def _setup_model(self) -> None:
        # 【完全复用标准库】动态切换观察空间以匹配 Transformer 隐藏层维度
        # 保存原始观察空间（用于初始化 ReplayBuffer）
        original_obs_space = self.observation_space
        
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        # 核心 Hook：临时切换到隐藏空间，使 Policy 初始化正确的输入维度
        self.observation_space = self.hidden_state_space
        
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
        # 还原原始观察空间，确保后续采样流程正常
        self.observation_space = original_obs_space

        self._convert_train_freq()

        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.state_transformer.train() # 开启 MAE 训练模式
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.actor_transformer.optimizer, self.critic_transformer.optimizer, self.transformer_optim]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        transformer_losses = []

        # 动态判断设备。只有 CUDA 模式才使用 GradScaler
        device_type = self.device.type
        use_amp = (device_type == "cuda")
        scaler = getattr(self, "scaler", None)
        if scaler is None and use_amp:
            self.scaler = th.amp.GradScaler('cuda')
            scaler = self.scaler

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # --- [Quant Expert] Input Noise Injection ---
            # 金融数据防过拟合关键：注入 0.5% 的高斯噪声，模拟市场微观结构噪声
            # 迫使模型学习分布特征而非记忆特定价格点
            noise = th.randn_like(replay_data.observations) * 0.005
            replay_data = replay_data._replace(observations=replay_data.observations + noise)
            # ------------------------------------------

            # We need to sample because 'log_std' may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # 批次融合在 CPU 上也能大幅减少算子调用开销
            combined_obs = th.cat([replay_data.observations, replay_data.next_observations], dim=0)
            
            # 动态适配设备类型，如果是 CPU 则自动禁用或使用 CPU 模式的 autocast
            with th.amp.autocast(device_type=device_type, enabled=use_amp):
                combined_out, temporal_short, temporal_long, combined_additional = self._state_transfer(combined_obs, mask_mode='mixed')
                
                state, next_state = th.chunk(combined_out, 2, dim=0)
                additional_feature, next_additional_feature = th.chunk(combined_additional, 2, dim=0)
                temporal_short_state, temporal_short_next = th.chunk(temporal_short, 2, dim=0)
                temporal_long_state, temporal_long_next = th.chunk(temporal_long, 2, dim=0)
                
                # state_for_actor = state.detach()
                # 使用 actor_alpha 动态控制回传给 MAE 的梯度
                state_for_actor = state * self.actor_alpha + state.detach() * (1 - self.actor_alpha)
                # 使用 critic_alpha 动态控制回传给 MAE 的梯度
                state_for_critic = state * self.critic_alpha + state.detach() * (1 - self.critic_alpha)
                
                combined_policy_input = th.cat([state_for_actor, next_state.detach()], dim=0)
                combined_additional_input = th.cat([additional_feature, next_additional_feature], dim=0)
                combined_temporal_short = th.cat([temporal_short_state, temporal_short_next], dim=0)
                combined_temporal_long = th.cat([temporal_long_state, temporal_long_next], dim=0)
                
                combined_policy_embed = self.actor_transformer(combined_policy_input, combined_temporal_short, combined_temporal_long, combined_additional_input)
                policy_embed, next_policy_embed = th.chunk(combined_policy_embed, 2, dim=0)

                actions_pi, log_prob = self.actor.action_log_prob(policy_embed)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None:
                    ent_coef = th.exp(self.log_ent_coef.detach()).clamp(min=0.001, max=0.5)
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor

                ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(ent_coef_loss).backward()
                    scaler.step(self.ent_coef_optimizer)
                else:
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

            # Target Q-values calculation
            with th.no_grad():
                with th.amp.autocast(device_type=device_type, enabled=use_amp):
                    next_actions, next_log_prob = self.actor.action_log_prob(next_policy_embed)
                    
                    next_critic_embed = self.critic_transformer_target(next_state.detach(), temporal_short_next, temporal_long_next, next_additional_feature)
                    next_q_values = th.cat(self.critic_target(next_critic_embed, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    # 【P0 改进】分段对数压缩：只对 |x| > 1000 的极端值进行压缩，保留大部分区域的线性特性
                    threshold = 1000.0
                    target_q_values = th.where(
                        th.abs(target_q_values) > threshold,
                        th.sign(target_q_values) * (1000 + (th.abs(target_q_values) - 1000)*0.1),
                        target_q_values
                    )
                    target_q_values = th.clamp(target_q_values, min=-5000, max=5000)


            # Optimize critic
            with th.amp.autocast(device_type=device_type, enabled=use_amp):
                #state:不带梯度控制的状态; state_for_critic: 带梯度控制的状态
                # current_critic_embed = self.critic_transformer(state, None, None, additional_feature)
                current_critic_embed = self.critic_transformer(state_for_critic, temporal_short_state, temporal_long_state, additional_feature)
                current_q_values = self.critic(current_critic_embed, replay_data.actions)
                critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            
            critic_losses.append(critic_loss.item())

            #重置 critic 和 critic Transformer, MAE 优化器的梯度，以确保它们只接收来自当前 critic_loss 的梯度
            self.critic.optimizer.zero_grad()
            self.critic_transformer.optimizer.zero_grad()
            self.transformer_optim.zero_grad() # 重置 MAE 优化器
            
            #更新 critic 和 critic Transformer 的参数，同时允许梯度流回 MAE 以进行微调
            if scaler is not None:
                scaler.scale(critic_loss).backward(retain_graph=True) # 保留计算图以供 Actor 微调 MAE
                
                # 【P0防过拟合】Critic梯度裁剪 (需先反缩放)
                scaler.unscale_(self.critic.optimizer)
                scaler.unscale_(self.critic_transformer.optimizer)
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                th.nn.utils.clip_grad_norm_(self.critic_transformer.parameters(), max_norm=1.0)
                # 注意：此处不裁剪 MAE，等待梯度累积完成后统一裁剪
                
                scaler.step(self.critic.optimizer)
                scaler.step(self.critic_transformer.optimizer)
                # scaler.step(self.transformer_optim) # 暂时不更新 MAE，等待梯度累积
            else:
                critic_loss.backward(retain_graph=True) # 保留计算图
                # 【P0防过拟合】Critic梯度裁剪
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                th.nn.utils.clip_grad_norm_(self.critic_transformer.parameters(), max_norm=1.0)
                # 注意：此处不裁剪 MAE，等待梯度累积完成后统一裁剪
                
                self.critic.optimizer.step()
                self.critic_transformer.optimizer.step()
                # self.transformer_optim.step() # 暂时不更新 MAE

            # 计算 actor loss
            with th.amp.autocast(device_type=device_type, enabled=use_amp):
                q_values_pi = th.cat(self.critic.forward(self.critic_transformer(state_for_actor, temporal_short_state, temporal_long_state, additional_feature), actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

                # log_prob 是 Agent 采取当前动作的概率对数。因为概率小于 1，所以 log_prob 是负数.
                # min_qf_pi 是双 Critic 网络对当前动作预估的Q值.
                # th.sum(actions, dim=-1)是对所有股票分配比例的总和（即总仓位,应该接近 1）的惩罚项.
                alpha = 0
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + alpha * th.abs(th.mean(th.sum(replay_data.actions, dim=-1))-1)

            actor_losses.append(actor_loss.item())

            #重置 Actor 和 Actor Transformer 的梯度，同时保留 MAE 优化器的梯度以供累积
            self.actor.optimizer.zero_grad()
            self.actor_transformer.optimizer.zero_grad()
            # self.transformer_optim.zero_grad() # 不要重置，因为要累积来自 Critic 的梯度
            
            #更新 Actor 和 Actor Transformer 的参数，同时允许梯度流回 MAE 以进行微调
            if scaler is not None:
                scaler.scale(actor_loss).backward()
                
                # 【P0防过拟合】Actor梯度裁剪 (需先反缩放)
                scaler.unscale_(self.actor.optimizer)
                scaler.unscale_(self.actor_transformer.optimizer)
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                th.nn.utils.clip_grad_norm_(self.actor_transformer.parameters(), max_norm=1.0)
                
                scaler.step(self.actor.optimizer)
                scaler.step(self.actor_transformer.optimizer)
                # scaler.step(self.transformer_optim) # 暂时不更新 MAE
                # scaler.update() # 移到最后统一步进
            else:
                actor_loss.backward()
                # 【P0防过拟合】Actor梯度裁剪
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                th.nn.utils.clip_grad_norm_(self.actor_transformer.parameters(), max_norm=1.0)
                self.actor.optimizer.step()
                self.actor_transformer.optimizer.step()
                # self.transformer_optim.step() # 暂时不更新 MAE
          
            # 最后统一更新 MAE 优化器 #暂时被注释掉
            if scaler is not None:
                # 【P0防过拟合】MAE梯度最终裁剪 (需先反缩放)
                # scaler.unscale_(self.transformer_optim)
                # th.nn.utils.clip_grad_norm_(self.state_transformer.parameters(), max_norm=2.0)
                
                # scaler.step(self.transformer_optim)
                # 在梯度步结束时必须调用 update()，否则下次 step() 会报错
                scaler.update()
            # else:
            #     # 【P0防过拟合】MAE梯度最终裁剪
            #     th.nn.utils.clip_grad_norm_(self.state_transformer.parameters(), max_norm=2.0)
            #     self.transformer_optim.step()

        # 更新目标网络 (Polyak Update)，这是 SAC 收敛的关键
        #if gradient_step % self.target_update_interval == 0:
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        polyak_update(self.critic_transformer.parameters(), self.critic_transformer_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        
        # 【P0监控】梯度健康度记录
        # if self._n_updates % 1000 == 0 and self.verbose > 0:
        if transformer_losses:
            self.logger.record("train/transformer_loss", np.mean(transformer_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        **kwargs,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            **kwargs,
        )

    def predict(
        self,
        test_obs: np.ndarray,
        deterministic: bool = False,
        state: np.ndarray = None,
        episode_start: np.ndarray = None,
    ) -> OffPolicyAlgorithm:

        flag = 0
        if len(test_obs.shape) == 2:
            test_obs = np.expand_dims(test_obs, axis=0)
            flag = 1

        # 保存当前训练模式，确保评估后恢复
        # 这是防御性编程，确保 predict 调用不会影响后续的训练
        was_training_state = self.state_transformer.training
        was_training_actor = self.actor_transformer.training

        self.state_transformer.eval()
        self.actor_transformer.eval()

        try:
            with th.no_grad():
                obs = th.FloatTensor(test_obs).to(self.transformer_device)
                obs_tensor, temporal_short, temporal_long, additional_feature = self._state_transfer(obs)
                state_tensor = self.actor_transformer(obs_tensor, temporal_short, temporal_long, additional_feature)
                obs_array = state_tensor.detach().cpu().numpy()

            if flag:
                obs_array = obs_array.squeeze(0)
            return super(SAC, self).predict(observation=obs_array, deterministic=deterministic, state=state, episode_start=episode_start)
        finally:
            # 确保恢复之前的训练模式
            if was_training_state:
                self.state_transformer.train()
            if was_training_actor:
                self.actor_transformer.train()

    def save_replay_buffer(self, path: Union[str, os.PathLike], max_save: Optional[int] = 10000) -> None:
        """
        保存 ReplayBuffer。使用 np.savez 存储
        """
        if self.replay_buffer is None:
            raise ValueError("The replay buffer is not defined.")

        pos = self.replay_buffer.pos
        full = self.replay_buffer.full
        buffer_size = self.replay_buffer.buffer_size
        
        # 这里限制最多导出条数
        current_count = buffer_size if full else pos
        n_to_save = min(current_count, max_save)

        if n_to_save <= 0:
            print("Buffer 里还没数据，不存了。")
            return

        # 定义保存路径（统一使用 .npz 格式）
        save_path = str(path)
        if not save_path.endswith('.npz'):
            save_path = os.path.splitext(save_path)[0] + ".npz"

        print(f"正在导出最新的 {n_to_save} 条数据至: {save_path} ...")
        
        # 获取最新 n_to_save 条数据的索引（处理环形缓冲区的回绕情况）
        if full:
            # 环形缓冲区中，最新数据在 pos 之前
            start_idx = (pos - n_to_save) % buffer_size
            if start_idx < pos:
                # 情况 A：最新数据连续，没有跨越缓冲区末尾
                idx = slice(start_idx, pos)
            else:
                # 情况 B：最新数据跨越了缓冲区末尾，需要拼接索引
                idx = np.concatenate([np.arange(start_idx, buffer_size), np.arange(0, pos)])
        else:
            # 未满时，最新数据就是末尾的 n_to_save 条
            idx = slice(pos - n_to_save, pos)

        # 准备数据字典（直接切片/索引）
        # 注意：导出的数据在加载时会被放在新 Buffer 的 0 到 n_to_save 位置
        save_dict = {
            "observations": self.replay_buffer.observations[idx],
            "next_observations": self.replay_buffer.next_observations[idx],  # 【修复】保存 next_observations
            "actions": self.replay_buffer.actions[idx],
            "rewards": self.replay_buffer.rewards[idx],
            "dones": self.replay_buffer.dones[idx],
            "pos": np.array([n_to_save]), # 加载后，下一个数据将从 n_to_save 开始存
            "full": np.array([False])      # 导出的子集通常视为未满状态
        }

        # 如果有 timeouts (SB3 默认会有)
        if hasattr(self.replay_buffer, "timeouts"):
            save_dict["timeouts"] = self.replay_buffer.timeouts[idx]

        # 核心优化：改用 savez 而非 savez_compressed，速度提升巨大
        np.savez(save_path, **save_dict)
        print(f"最新 {n_to_save} 条 Buffer 数据保存成功！")

    def load_replay_buffer(self, path: Union[str, os.PathLike], max_load: Optional[int] = None) -> None:
        """
        加载 ReplayBuffer。通过 mmap_mode 实现从磁盘到预分配空间的“串行写入”。
        增加了 max_load 参数，用于指定加载多少条数据。
        """
        if self.replay_buffer is None:
            raise ValueError("The replay buffer is not defined.")

        # 统一使用 .npz 格式
        path_str = str(path)
        if not path_str.endswith('.npz'):
            path_str = os.path.splitext(path_str)[0] + ".npz"

        if not os.path.exists(path_str):
            raise FileNotFoundError(f"老大，找不到优化的 Buffer 文件: {path_str}")

        print(f"正在使用磁盘映射模式（mmap）从 {path_str} 串行载入数据...")
        
        # 核心：使用 mmap_mode='r'。这不会把数组读入 RAM，而是直接映射磁盘文件
        with np.load(path_str, mmap_mode='r') as data:
            loaded_pos = int(data["pos"][0])
            # loaded_full = bool(data["full"][0])
            
            # 老大，如果指定了 max_load，则只加载文件里最后（最新）的这么多条数据
            n_to_load = loaded_pos
            if max_load is not None:
                n_to_load = min(loaded_pos, max_load)
            
            # 【修复】老大，这里必须限制 n_to_load 不得超过当前 Buffer 的容量，否则切片赋值会报错
            n_to_load = min(n_to_load, self.replay_buffer.buffer_size)
            
            # 计算起始索引，确保从文件末尾抓取最新的数据
            start_idx = loaded_pos - n_to_load

            # 串行写入：直接将映射的磁盘数据拷贝到预分配的内存空间中
            # CPU 会在这一步执行高效的 Block Copy，不占用额外中转内存
            print(f"正在将文件中最新的 {n_to_load} 条经验（索引 {start_idx} 到 {loaded_pos}）串行拷贝至预分配内存...")
            
            # 处理 Observation (兼容 SB3 的 (n, 1, ...) 结构)
            source_obs = data["observations"]
            if len(source_obs.shape) == len(self.replay_buffer.observations.shape):
                self.replay_buffer.observations[:n_to_load] = source_obs[start_idx:loaded_pos]
            else:
                self.replay_buffer.observations[:n_to_load, 0] = source_obs[start_idx:loaded_pos]

            # 【修复】加载 next_observations
            if "next_observations" in data:
                source_next_obs = data["next_observations"]
                if len(source_next_obs.shape) == len(self.replay_buffer.next_observations.shape):
                    self.replay_buffer.next_observations[:n_to_load] = source_next_obs[start_idx:loaded_pos]
                else:
                    self.replay_buffer.next_observations[:n_to_load, 0] = source_next_obs[start_idx:loaded_pos]
            else:
                # 兼容旧版 buffer 文件（没有 next_observations）
                print("警告：旧版 Buffer 文件缺少 next_observations，训练可能异常，建议重新收集数据")

            self.replay_buffer.actions[:n_to_load] = data["actions"][start_idx:loaded_pos]
            self.replay_buffer.rewards[:n_to_load] = data["rewards"][start_idx:loaded_pos]
            self.replay_buffer.dones[:n_to_load] = data["dones"][start_idx:loaded_pos]
            
            if "timeouts" in data and hasattr(self.replay_buffer, "timeouts"):
                self.replay_buffer.timeouts[:n_to_load] = data["timeouts"][start_idx:loaded_pos]
            
            # 【修复】pos 必须取模，如果加载了 buffer_size 条数据，pos 应该回绕到 0
            self.replay_buffer.pos = n_to_load % self.replay_buffer.buffer_size
            # 如果加载的数据量达到了 buffer_size，则标记为已满
            self.replay_buffer.full = n_to_load >= self.replay_buffer.buffer_size
            
        print(f"串行载入完成！当前 Buffer 状态: {'已满' if self.replay_buffer.full else '未满'}, 位置: {self.replay_buffer.pos}")

    # def set_parameters(self, load_path_or_dict, exact_match: bool = True, device: Union[th.device, str] = "auto") -> None:
    #     """
    #     重置参数加载逻辑，使其在 A 轮转 B 轮时能够跳过缺失的新参数。
    #     """
    #     # 老大，我们将 exact_match 强制设为 False，这样如果 zip 里没找到新定义的优化器或 Target 网络，
    #     # 它会跳过并打印警告，而不是直接报错崩溃。
    #     super().set_parameters(load_path_or_dict, exact_match=False, device=device)

    # 定在保存模型时需要排除（不保存）的参数名称列表
    # 这些组件的实际权重已经通过 _get_torch_save_params() 方法单独保存为 PyTorch state_dict 格式
    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target", "critic_transformer_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        # 保存基础 SAC 组件
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        # 保存 entropy coefficient 相关
        saved_pytorch_variables = ["log_ent_coef"]  # 无论是否学习 ent_coef，都保存 log_ent_coef 以保持接口一致
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        # 保存 SAC_MAE 特有的 Transformer 组件
        # state_transformer: Transformer 模型及其优化器
        state_dicts.extend(["state_transformer", "transformer_optim"])

        # 【修复】使用提升后的属性名，并增加对目标网络的保存
        state_dicts.extend(["actor_transformer", "actor_transformer_optim"])
        state_dicts.extend(["critic_transformer", "critic_transformer_optim"])
        state_dicts.extend(["critic_transformer_target"])

        return state_dicts, saved_pytorch_variables

    def _state_transfer(self, x, seed=None, mask_mode='nope'):
        """
        状态转换方法，使用 MAE Transformer 进行状态编码
        为每个模式应用其最快速的实现

        :param x: 输入状态 [bs, stock_num, features]
        :param seed: 可选的随机种子，用于生成 mask。如果提供确保相同的 seed 生成相同的 mask
        :param mask_mode: mask 模式，可选值：
            - 'stock': 屏蔽股票（默认），随机选择部分股票，屏蔽其全部特征
            - 'feature': 屏蔽技术指标，随机选择部分特征，对所有股票屏蔽这些特征
            - 'mixed': 混合模式，同时随机屏蔽部分股票和部分特征
            - 'nope': 不进行任何屏蔽，保留所有特征和股票
        :return: 编码后的状态、时序特征、附加特征、重建损失
        """
        bs, stock_num = x.shape[0], x.shape[1]
        feat_dim = self.in_feat  # 特征维度 (96，stock_num + tech_dim)

        batch_enc1 = x[:, :, :feat_dim]  # [bs, stock_num, feat_dim] 包含 cov+technical_list

        if mask_mode == 'stock':
            # ==================== 模式1: 屏蔽股票 - 最快版 ====================
            # 随机选择部分股票，屏蔽其全部特征
            mask = th.ones_like(batch_enc1)

            num_mask = max(1, int(stock_num * 0.2))
            if seed is not None:
                with th.random.fork_rng():
                    th.random.manual_seed(seed)
                    _, mask_stock_indices = th.rand(bs, stock_num, device=x.device).topk(num_mask, dim=-1, largest=False)
            else:
                _, mask_stock_indices = th.rand(bs, stock_num, device=x.device).topk(num_mask, dim=-1, largest=False)  # [bs, num_mask]

            # 使用向量化操作屏蔽选中的股票（所有特征）
            stock_mask = th.ones(bs, stock_num, device=x.device)  # [bs, stock_num]
            stock_mask.scatter_(1, mask_stock_indices, 0)  # 将选中的股票位置置 0
            # 扩展到所有特征: [bs, stock_num] -> [bs, stock_num, feat_dim]
            mask = stock_mask.unsqueeze(2).expand(-1, -1, feat_dim)

            enc_inp = mask * batch_enc1
            
            # 只运行 Encoder 部分，跳过 Decoder
            # 中间层特征enc_out真正用于actor和critic(shape: [Batch_size, Stock_num, hidden_out])
            # 重建结果output只用于评估重建损失( Shape: [Batch_size, Stock_num, c_out_construction])
            # enc_out, _, output = self.state_transformer(enc_inp, enc_inp)
            enc_out = self.state_transformer.enc_embedding(enc_inp)
            enc_out, _ = self.state_transformer.encoder(enc_out)

        elif mask_mode == 'feature':
            # ==================== 模式2: 屏蔽技术指标 - 最快版 ====================
            # 随机选择部分特征，对所有股票屏蔽这些特征
            num_mask = max(1, int(feat_dim * 0.1))
            if seed is not None:
                with th.random.fork_rng():
                    th.random.manual_seed(seed)
                    _, mask_feat_indices = th.rand(bs, feat_dim, device=x.device).topk(num_mask, dim=-1, largest=False)
            else:
                _, mask_feat_indices = th.rand(bs, feat_dim, device=x.device).topk(num_mask, dim=-1, largest=False)  # [bs, num_mask]

            # 使用向量化操作屏蔽选中的特征（对所有股票生效）
            feat_mask = th.ones(bs, feat_dim, device=x.device)  # [bs, feat_dim]
            feat_mask.scatter_(1, mask_feat_indices, 0)  # 将选中的特征位置置 0
            # 扩展到所有股票: [bs, feat_dim] -> [bs, stock_num, feat_dim]
            mask = feat_mask.unsqueeze(1).expand(-1, stock_num, -1)

            enc_inp = mask * batch_enc1
            
            # 只运行 Encoder 部分
            enc_out = self.state_transformer.enc_embedding(enc_inp)
            enc_out, _ = self.state_transformer.encoder(enc_out)

        elif mask_mode == 'mixed':
            # ==================== 模式3: 混合模式，同时屏蔽股票和特征  - 优化版 ====================
            num_stock_mask = max(1, int(stock_num * 0.2))
            num_feat_mask = max(1, int(feat_dim * 0.1))
            if seed is not None:
                with th.random.fork_rng():
                    th.random.manual_seed(seed)
                    _, mask_stock_indices = th.rand(bs, stock_num, device=x.device).topk(num_stock_mask, dim=-1, largest=False)
                    _, mask_feat_indices = th.rand(bs, feat_dim, device=x.device).topk(num_feat_mask, dim=-1, largest=False)
            else:
                _, mask_stock_indices = th.rand(bs, stock_num, device=x.device).topk(num_stock_mask, dim=-1, largest=False)
                _, mask_feat_indices = th.rand(bs, feat_dim, device=x.device).topk(num_feat_mask, dim=-1, largest=False)

            # 创建股票mask: [bs, stock_num] -> [bs, stock_num, feat_dim]
            stock_mask = th.ones(bs, stock_num, device=x.device)  # [bs, stock_num]
            stock_mask.scatter_(1, mask_stock_indices, 0)  # 将选中的股票位置置 0
            stock_mask_expanded = stock_mask.unsqueeze(2).expand(-1, -1, feat_dim)

            # 创建特征mask: [bs, feat_dim] -> [bs, stock_num, feat_dim]
            feat_mask = th.ones(bs, feat_dim, device=x.device)  # [bs, feat_dim]
            feat_mask.scatter_(1, mask_feat_indices, 0)  # 将选中的特征位置置 0
            feat_mask_expanded = feat_mask.unsqueeze(1).expand(-1, stock_num, -1)

            # 组合mask：两个mask相乘（只有未被任一mask屏蔽的位置才保留）
            mask = stock_mask_expanded * feat_mask_expanded

            enc_inp = mask * batch_enc1
            
            # 只运行 Encoder 部分
            enc_out = self.state_transformer.enc_embedding(enc_inp)
            enc_out, _ = self.state_transformer.encoder(enc_out)

        else:  # if mask_mode == 'nope':
            # ==================== 模式4: 不屏蔽任何特征或股票 ====================
            enc_inp = batch_enc1
            # 只运行 Encoder 部分，跳过 Decoder
            # 中间层特征enc_out真正用于actor和critic(shape: [Batch_size, Stock_num, hidden_out])
            # 重建结果output只用于评估重建损失( Shape: [Batch_size, Stock_num, c_out_construction])
            # enc_out, _, output = self.state_transformer(enc_inp, enc_inp)
            enc_out = self.state_transformer.enc_embedding(enc_inp)
            enc_out, _ = self.state_transformer.encoder(enc_out)

        #temporal_feature_short = None
        #temporal_feature_long = None
        # 使用环境对应的隐藏维度进行切片，而非 MAE 的 hidden_out (防止维度不一致导致切片错位)
        env_hidden_dim = self.env_hidden_dim
        temporal_feature_short = x[:, :, feat_dim: env_hidden_dim + feat_dim]
        temporal_feature_long = x[:, :, env_hidden_dim + feat_dim: env_hidden_dim * 2 + feat_dim]

        # 【精准特征切片：仅包含技术指标和日期】
        # 1. 提取技术指标 (位于协方差矩阵之后)
        # x 结构: [Cov (stock_num)] [Tech (feat_dim - stock_num)][hidden_out][Date (12)][Holding Ratio (1)]
        tech_features = x[:, :, stock_num : feat_dim] 
        # 2. 提取日期特征及其他附加特征 (在时序特征之后)
        # 这里包含了 date_features (12) 和 holding_assets_ratio (1)
        tail_features = x[:, :, feat_dim + env_hidden_dim * 2:]        
        # date_features = x[:, :, feat_dim+hidden_out*2:]        
        # 3. 合并为纯净的 additional_feature (排除协方差数据)
        additional_feature = th.cat((tech_features, tail_features), dim=-1)

        #各元素维度：[bs, stock_num, hidden_out], [bs, stock_num, hidden_out], [bs, stock_num, hidden_out],
        # [bs, stock_num, x.shape[-1] - feat_dim - hidden_out*2]
        return enc_out, temporal_feature_short, temporal_feature_long, additional_feature

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        tensorboard_log: Optional[str] = None,
        load_optimizer: bool = True,
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ):
        """
        加载模型时添加 load_optimizer 参数控制是否加载优化器状态
        
        :param path: 模型文件路径
        :param env: 环境对象（可选）
        :param tensorboard_log: TensorBoard 日志目录
        :param load_optimizer: 是否加载优化器状态，默认为 True
        :param custom_objects: 自定义对象字典，用于替换加载时的对象
        :param print_system_info: 是否打印系统信息
        :param device: 设备类型
        :param kwargs: 其他关键字参数
        :return: 加载后的模型实例
        """
        
        # 从 zip 文件加载数据
        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects, print_system_info=print_system_info
        )
        
        # 创建模型实例（不初始化模型）
        model = cls(
            policy=data["policy_class"],
            env=env,
            tensorboard_log=tensorboard_log,
            device=device,
            _init_setup_model=False,
            **kwargs,
        )
        
        # 更新模型属性
        model.__dict__.update(data)
        
        # 重新设置模型（创建网络和优化器）
        model._setup_model()
        
        # 加载参数（包括优化器状态）
        # 如果 load_optimizer=False，则跳过优化器状态的加载
        if load_optimizer:
            model.set_parameters(params, exact_match=True, device=device)
        else:
            # 只加载非优化器的参数（模型权重等）
            for name in params:
                if "optimizer" not in name.lower():
                    attr = recursive_getattr(model, name)
                    if isinstance(attr, th.optim.Optimizer):
                        continue  # 跳过优化器
                    else:
                        attr.load_state_dict(params[name], strict=True)
        
        # 加载 PyTorch 变量
        if pytorch_variables is not None:
            for name in pytorch_variables:
                attr = recursive_getattr(model, name)
                # Tensor 类型直接赋值，nn.Module 类型调用 load_state_dict
                if isinstance(attr, th.Tensor):
                    attr.data = pytorch_variables[name].data.to(device=attr.device)
                else:
                    attr.load_state_dict(pytorch_variables[name])

        return model