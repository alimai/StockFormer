from typing import Any, Dict, List, Optional, Tuple, Type, Union
import random

import gymnasium as gym
import numpy as np
import torch as th
from torch.nn import functional as F
from collections import OrderedDict


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
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
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
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        dropout=0.05,
        transformer_device = None,
        transformer_path = None,
        actor_alpha=1.0,
        critic_alpha=1.0,
    ):
        # 【关键修复】在 super().__init__ 之前获取并设置隐藏状态空间
        # 否则父类初始化过程中调用 _setup_model 时会因找不到 hidden_state_space 报错
        self.hidden_state_space = None
        if hasattr(env, "hidden_state_space"):
            self.hidden_state_space = env.hidden_state_space
        elif hasattr(env, "get_attr"):
            try:
                self.hidden_state_space = env.get_attr("hidden_state_space")[0]
            except Exception:
                pass
        
        if self.hidden_state_space is None and hasattr(self, "env") and self.env is not None:
            if hasattr(self.env, "get_attr"):
                try:
                    self.hidden_state_space = self.env.get_attr("hidden_state_space")[0]
                except Exception:
                    pass

        super(SAC, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,#使用_update_learning_rate()后学习速率真正被赋初值的地方
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
                                             d_model=d_model, d_ff=d_ff, dropout=dropout).to(transformer_device)

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
        self.transformer_optim = th.optim.Adam(self.state_transformer.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.transformer_criteria = th.nn.MSELoss()

        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha


        # 获取环境的时序特征隐藏维度 (即 prediction model 的 hidden_channel)
        if hasattr(env, "hidden_channel"):
            self.env_hidden_dim = env.hidden_channel
        elif hasattr(env, "get_attr"):
            try:
                self.env_hidden_dim = env.get_attr("hidden_channel")[0]
            except Exception:
                self.env_hidden_dim = d_model
        else:
            self.env_hidden_dim = d_model

        # 向 Policy Transformer 传递附加信号维度(Tech + Date)
        # self.in_feat (enc_in) = stock_num + tech_dim
        # additional_dim = self.in_feat - stock_num + 12(Tech + Date)
        #additional_dim = self.hidden_state_space.shape[1] - d_model
        stock_num = env.observation_space.shape[0]
        # 使用环境隐藏维度进行计算，确保与环境生成的 Observation 结构对齐
        additional_dim = env.observation_space.shape[1] - stock_num  - self.env_hidden_dim * 2
        # additional_dim = env.observation_space.shape[1] - stock_num  - d_model* 2
        
        self.actor_transformer = policy_transformer_attn2(d_model=d_model, dropout=dropout, lr=learning_rate, device=transformer_device, additional_dim=additional_dim).to(transformer_device)
        self.critic_transformer = policy_transformer_attn2(d_model=d_model, dropout=dropout, lr=learning_rate, device=transformer_device, additional_dim=additional_dim).to(transformer_device)

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

        # 老大，动态判断设备。只有 CUDA 模式才使用 GradScaler
        device_type = self.device.type
        use_amp = (device_type == "cuda")
        scaler = getattr(self, "scaler", None)
        if scaler is None and use_amp:
            self.scaler = th.amp.GradScaler('cuda')
            scaler = self.scaler

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # 批次融合在 CPU 上也能大幅减少算子调用开销
            combined_obs = th.cat([replay_data.observations, replay_data.next_observations], dim=0)
            seed = random.randint(0, 2**31 - 1)
            
            # 只有在最后一步或特定间隔才计算 reconstruction loss
            should_compute_loss = ((gradient_step+1)%(gradient_steps//5)==0)
            
            # 动态适配设备类型，如果是 CPU 则自动禁用或使用 CPU 模式的 autocast
            with th.amp.autocast(device_type=device_type, enabled=use_amp):
                combined_out, temporal_short, temporal_long, combined_additional, combined_loss = self._state_transfer(
                    combined_obs, seed=seed, mask_mode='mixed', compute_loss=should_compute_loss)
                
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
                    ent_coef = th.exp(self.log_ent_coef.detach())
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
                    
                    next_critic_embed = self.critic_transformer(next_state.detach(), temporal_short_next, temporal_long_next, next_additional_feature)
                    next_q_values = th.cat(self.critic_target(next_critic_embed, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Optimize critic
            with th.amp.autocast(device_type=device_type, enabled=use_amp):
                #state:不带梯度控制的状态; state_for_critic: 带梯度控制的状态
                # current_critic_embed = self.critic_transformer(state, None, None, additional_feature)
                current_critic_embed = self.critic_transformer(state_for_critic, temporal_short_state, temporal_long_state, additional_feature)
                current_q_values = self.critic(current_critic_embed, replay_data.actions)
                critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            self.critic_transformer.optimizer.zero_grad()
            self.transformer_optim.zero_grad() # 重置 MAE 优化器
            
            if scaler is not None:
                scaler.scale(critic_loss).backward(retain_graph=True) # 保留计算图以供 Actor 和 MAE 更新
                scaler.step(self.critic.optimizer)
                scaler.step(self.critic_transformer.optimizer)
                # scaler.step(self.transformer_optim) # 暂时不更新 MAE，等待梯度累积
            else:
                critic_loss.backward(retain_graph=True) # 保留计算图
                self.critic.optimizer.step()
                self.critic_transformer.optimizer.step()
                # self.transformer_optim.step() # 暂时不更新 MAE

            # Optimize actor
            with th.amp.autocast(device_type=device_type, enabled=use_amp):
                q_values_pi = th.cat(self.critic.forward(self.critic_transformer(state_for_actor, temporal_short_state, temporal_long_state, additional_feature), actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

                alpha = 0
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + alpha * th.abs(th.mean(th.sum(replay_data.actions, dim=-1))-1)

            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            self.actor_transformer.optimizer.zero_grad()
            # self.transformer_optim.zero_grad() # 不要重置，因为要累积来自 Critic 的梯度
            
            if scaler is not None:
                scaler.scale(actor_loss).backward(retain_graph=should_compute_loss) # 如果有重建损失则保留计算图
                scaler.step(self.actor.optimizer)
                scaler.step(self.actor_transformer.optimizer)
                # scaler.step(self.transformer_optim) # 暂时不更新 MAE
                # 在每个梯度步结束时必须调用 update()，否则下次 step() 会报错
                # scaler.update() # 移到最后统一步进
            else:
                actor_loss.backward(retain_graph=should_compute_loss) # 如果有重建损失则保留计算图
                self.actor.optimizer.step()
                self.actor_transformer.optimizer.step()
                # self.transformer_optim.step() # 暂时不更新 MAE

            if should_compute_loss:
                # 正式更新 MAE 模型（自监督部分）
                # self.transformer_optim.zero_grad() # 不要重置，累积之前的梯度
                if scaler is not None:
                    scaler.scale(combined_loss).backward()
                    # scaler.step(self.transformer_optim)
                    # scaler.update()
                else:
                    combined_loss.backward()
                    # self.transformer_optim.step()
                transformer_losses.append(combined_loss.item())
            
            # 老大，最后统一步进 MAE 优化器，避免 inplace 错误
            if scaler is not None:
                scaler.step(self.transformer_optim)
                scaler.update()
            else:
                self.transformer_optim.step()

            # # 老大，更新目标网络 (Polyak Update)，这是 SAC 收敛的关键
            # if gradient_step % self.target_update_interval == 0:
            #     polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
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
    ) -> OffPolicyAlgorithm:

        flag = 0
        if len(test_obs.shape) == 2:
            test_obs = np.expand_dims(test_obs, axis=0)
            flag = 1

        # 保存当前训练模式，确保评估后恢复
        # 这是防御性编程，确保 predict 调用不会影响后续的训练
        was_training = self.state_transformer.training
        self.state_transformer.eval()
        try:
            with th.no_grad():
                obs = th.FloatTensor(test_obs).to(self.transformer_device)
                obs_tensor, temporal_short, temporal_long, additional_feature,_ = self._state_transfer(obs, compute_loss=False)
                state_tensor = self.actor_transformer(obs_tensor, temporal_short, temporal_long, additional_feature)
                obs_array = state_tensor.detach().cpu().numpy()

            if flag:
                obs_array = obs_array.squeeze(0)
            return super(SAC, self).predict(observation=obs_array, deterministic=deterministic)
        finally:
            # 确保恢复之前的训练模式
            if was_training:
                self.state_transformer.train()

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        # 保存基础 SAC 组件
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]

        # 保存 entropy coefficient 相关
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]

        # 保存 SAC_MAE 特有的 Transformer 组件
        # state_transformer: Transformer 模型及其优化器
        state_dicts.extend(["state_transformer", "transformer_optim"])

        # actor_transformer 和 critic_transformer: 每个都有内部的 optimizer
        state_dicts.extend(["actor_transformer", "actor_transformer.optimizer"])
        state_dicts.extend(["critic_transformer", "critic_transformer.optimizer"])

        return state_dicts, saved_pytorch_variables

    def _state_transfer(self, x, seed=None, mask_mode='nope', compute_loss=True):
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
        :param compute_loss: 是否计算重建损失。如果不计算，将跳过 Decoder 以加速。
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
            
            # 按需计算 Decoder 部分，提速核心
            if compute_loss:
                # 重建结果output只用于评估重建损失( Shape: [Batch_size, Stock_num, c_out_construction])
                # 中间层特征enc_out真正用于actor和critic(shape: [Batch_size, Stock_num, d_model])
                enc_out, _, output = self.state_transformer(enc_inp, enc_inp)

                # 计算被屏蔽股票的重建损失
                # mask_stock_indices: [bs, num_mask] -> 扩展为 [bs, num_mask, feat_dim]
                gather_idx = mask_stock_indices.unsqueeze(2).expand(-1, -1, feat_dim)  # [bs, num_mask, feat_dim]

                pred = th.gather(output, 1, gather_idx)  # [bs, num_mask, feat_dim]
                true = th.gather(batch_enc1, 1, gather_idx)  # [bs, num_mask, feat_dim]
                loss = self.transformer_criteria(pred, true)
            else:
                # 只运行 Encoder 部分，跳过 Decoder
                enc_out = self.state_transformer.enc_embedding(enc_inp)
                enc_out, _ = self.state_transformer.encoder(enc_out)
                loss = th.tensor(0.0, device=x.device)

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
            
            if compute_loss:
                enc_out, _, output = self.state_transformer(enc_inp, enc_inp)

                # 计算被屏蔽特征的重建损失
                # mask_feat_indices: [bs, num_mask] -> 扩展为 [bs, stock_num, num_mask]
                gather_idx = mask_feat_indices.unsqueeze(1).expand(-1, stock_num, -1)  # [bs, stock_num, num_mask]

                pred = th.gather(output, 2, gather_idx)  # [bs, stock_num, num_mask]
                true = th.gather(batch_enc1, 2, gather_idx)  # [bs, stock_num, num_mask]
                loss = self.transformer_criteria(pred, true)
            else:
                # 只运行 Encoder 部分
                enc_out = self.state_transformer.enc_embedding(enc_inp)
                enc_out, _ = self.state_transformer.encoder(enc_out)
                loss = th.tensor(0.0, device=x.device)

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
            
            if compute_loss:
                enc_out, _, output = self.state_transformer(enc_inp, enc_inp)

                # 计算被屏蔽股票的重建损失
                # mask_stock_indices: [bs, num_stock_mask] -> 扩展为 [bs, num_stock_mask, feat_dim]
                stock_gather_idx = mask_stock_indices.unsqueeze(2).expand(-1, -1, feat_dim)  # [bs, num_stock_mask, feat_dim]
                stock_pred = th.gather(output, 1, stock_gather_idx)  # [bs, num_stock_mask, feat_dim]
                stock_true = th.gather(batch_enc1, 1, stock_gather_idx)  # [bs, num_stock_mask, feat_dim]

                # 计算被屏蔽特征的重建损失
                # mask_feat_indices: [bs, num_feat_mask] -> 扩展为 [bs, stock_num, num_feat_mask]
                feat_gather_idx = mask_feat_indices.unsqueeze(1).expand(-1, stock_num, -1)  # [bs, stock_num, num_feat_mask]
                feat_pred = th.gather(output, 2, feat_gather_idx)  # [bs, stock_num, num_feat_mask]
                feat_true = th.gather(batch_enc1, 2, feat_gather_idx)  # [bs, stock_num, num_feat_mask]

                # 组合损失：股票mask损失 + 特征mask损失
                # 优化：使用flatten替代reshape以提高性能
                pred = th.cat([stock_pred.flatten(), feat_pred.flatten()], dim=0)
                true = th.cat([stock_true.flatten(), feat_true.flatten()], dim=0)
                loss = self.transformer_criteria(pred, true)
            else:
                # 只运行 Encoder 部分
                enc_out = self.state_transformer.enc_embedding(enc_inp)
                enc_out, _ = self.state_transformer.encoder(enc_out)
                loss = th.tensor(0.0, device=x.device)

        else:  # if mask_mode == 'nope':
            # ==================== 模式4: 不屏蔽任何特征或股票 ====================
            enc_inp = batch_enc1
            if compute_loss:
                enc_out, _, output = self.state_transformer(enc_inp, enc_inp)
                # 由于没有进行掩码，无法计算重建损失，返回零损失
                loss = th.tensor(0.0, device=x.device)
            else:
                # 只运行 Encoder 部分
                enc_out = self.state_transformer.enc_embedding(enc_inp)
                enc_out, _ = self.state_transformer.encoder(enc_out)
                loss = th.tensor(0.0, device=x.device)

        #temporal_feature_short = None
        #temporal_feature_long = None
        # 使用环境对应的隐藏维度进行切片，而非 MAE 的 d_model (防止维度不一致导致切片错位)
        env_hidden_dim = self.env_hidden_dim
        # hidden_channel = enc_out.shape[-1]
        temporal_feature_short = x[:, :, feat_dim: env_hidden_dim + feat_dim]
        # temporal_feature_short = x[:, :, feat_dim: hidden_channel+feat_dim]
        temporal_feature_long = x[:, :, env_hidden_dim + feat_dim: env_hidden_dim * 2 + feat_dim]
        # temporal_feature_long = x[:, :, hidden_channel+feat_dim: hidden_channel*2+feat_dim]

        # 【精准特征切片：仅包含技术指标和日期】
        # 1. 提取技术指标 (位于协方差矩阵之后)
        # x 结构: [Cov (stock_num)] [Tech (feat_dim - stock_num)][hidden_channel][Date (12)]
        tech_features = x[:, :, stock_num : feat_dim] 
        # 2. 提取日期特征 (在时序特征之后)
        date_features = x[:, :, feat_dim + env_hidden_dim * 2:]        
        # date_features = x[:, :, feat_dim+hidden_channel*2:]        
        # 3. 合并为纯净的 additional_feature (排除协方差数据)
        additional_feature = th.cat((tech_features, date_features), dim=-1)

        #各元素维度：[bs, stock_num, d_model], [bs, stock_num, hidden_channel], [bs, stock_num, hidden_channel],
        # [bs, stock_num, x.shape[-1] - feat_dim - hidden_channel*2], loss (标量)
        return enc_out, temporal_feature_short, temporal_feature_long, additional_feature, loss