from typing import Any, Dict, List, Optional, Tuple, Type, Union
import random

import gym
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
import pdb


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

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
        critic_alpha=1,
        actor_alpha=0,
    ):

        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
        )
        
        # 【关键修改】跨越包装器获取隐藏状态空间
        # 官方 SB3 包装器不会直接透传自定义属性，需要通过 get_attr 获取
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

        if transformer_path is not None:
            state_dict = th.load(transformer_path, map_location=transformer_device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            self.state_transformer.load_state_dict(new_state_dict)
            print("Successfully load pretrained model...", transformer_path)
        else:
            print("Successfully initialize transformer model...")

        self.transformer_device = transformer_device
        self.transformer_optim = th.optim.Adam(self.state_transformer.parameters(), lr=1e-5)
        self.transformer_criteria = th.nn.MSELoss()

        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha


        self.actor_transformer = policy_transformer_attn2(d_model=d_model, dropout=dropout, lr=learning_rate, device=transformer_device).to(transformer_device)
        self.critic_transformer = policy_transformer_attn2(d_model=d_model, dropout=dropout, lr=learning_rate, device=transformer_device).to(transformer_device)

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
        self.state_transformer.train()
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.actor_transformer.optimizer, self.critic_transformer.optimizer, self.transformer_optim]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        transformer_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            # pdb.set_trace()
            # 【论文一致性修复】根据论文 Table 7 消融实验：
            # - Critic 梯度需要传播到 relation inference module（state_transformer）
            # - Actor 梯度不需要传播到 relation inference module
            # 使用随机数作为种子，确保同一个 batch 的 state 和 next_state 使用相同的 mask 模式
            seed = random.randint(0, 2**31 - 1)
            # mask_mode 控制屏蔽方式: 'stock'(屏蔽股票) 或 'feature'(屏蔽技术指标) 或 'mixed' 或 'nope'(默认，不屏蔽)
            state, temporal_feature_short, temporal_feature_long, additional_feature, loss_s = self._state_transfer(
                replay_data.observations, seed=seed)#,mask_mode = 'feature'
            # 【论文一致性】Actor 使用 detach 后的 state，防止 Actor 梯度传播到 state_transformer
            state_for_actor = state.detach()
            actions_pi, log_prob = self.actor.action_log_prob(self.actor_transformer(state_for_actor, temporal_feature_short, temporal_feature_long, additional_feature))
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # pdb.set_trace()
            # 计算 next_state 时使用 detach，确保状态表示稳定
            # 这样可以避免 state_transformer 更新导致的状态表示突然变化影响 target_q_values
            # 使用相同的随机 seed，确保 state 和 next_state 使用相同的 mask 模式
            # 这样可以避免随机 mask 导致的状态表示不一致，从而减少 critic_loss 的异常峰值            
            # mask_mode 控制屏蔽方式: 'stock'(屏蔽股票) 或 'feature'(屏蔽技术指标) 或 'mixed' 或 'nope'(默认，不屏蔽)
            next_state, next_temporal_feature_short, next_temporal_feature_long, next_additional_feature, loss_ns = self._state_transfer(
                replay_data.next_observations, seed=seed)
            # 使用 detach 确保状态表示稳定，避免 state_transformer 更新影响 target 计算
            next_state = next_state.detach()
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(self.actor_transformer(next_state, next_temporal_feature_short, next_temporal_feature_long, next_additional_feature))

                # next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(self.critic_transformer(next_state, next_temporal_feature_short, next_temporal_feature_long, next_additional_feature), next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            # 【论文一致性】Critic 使用原始 state（不 detach），允许梯度传播到 state_transformer
            current_q_values = self.critic(self.critic_transformer(state, temporal_feature_short, temporal_feature_long, additional_feature), replay_data.actions)

            # Compute critic loss
            # pdb.set_trace() # get critic loss item value
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            # 调试：处理异常大的critic_loss
            # loss_gate = 100000.0*np.mean(ent_coefs)+1000
            # if critic_loss.item() > loss_gate:  # 阈值设为50
            #     print(f"WARNING: Large critic_loss at step {self.num_timesteps}: {critic_loss.item():.4f}")
            #     print(f"  current_q_values range: {current_q_values[0].min().item():.4f} to {current_q_values[0].max().item():.4f}")
            #     print(f"  next_q_values range: {next_q_values.min().item():.4f} to {next_q_values.max().item():.4f}")
            #     print(f"  replay_data.rewards range: {replay_data.rewards.min().item():.4f} to {replay_data.rewards.max().item():.4f}")
            #     print(f"  target_q_values range: {target_q_values.min().item():.4f} to {target_q_values.max().item():.4f}")
            #     print(f"  done ratio: {replay_data.dones.float().mean().item():.3f}")
            #     # 检查是否有NaN或Inf
            #     if th.isnan(critic_loss) or th.isinf(critic_loss):
            #         print("  CRITICAL: NaN or Inf detected in critic_loss!")
            #         print()
            #     # 裁剪 critic_loss 值以防止发散
            #     critic_loss = th.clamp(critic_loss, min=-loss_gate, max=loss_gate)
            critic_losses.append(critic_loss.item())

            # 检查 replay_data.rewards 最大值是否大于50
            if replay_data.rewards.max().item() > 50.0:
                print(f"WARNING: Large reward detected at step {self.num_timesteps}")
                print(f"  replay_data.rewards max: {replay_data.rewards.max().item():.4f}")
                print(f"  replay_data.rewards range: {replay_data.rewards.min().item():.4f} to {replay_data.rewards.max().item():.4f}")
                print(f"  done ratio: {replay_data.dones.float().mean().item():.3f}")
                print()

            # pdb.set_trace()
            # 【论文一致性】Optimize the critic，同时更新 state_transformer（relation inference module）
            # 根据论文："propagates the analytic gradients of state values back into the relation inference module"
            self.critic.optimizer.zero_grad()
            self.critic_transformer.optimizer.zero_grad()
            self.transformer_optim.zero_grad()  # 【论文一致性】包含 state_transformer
            critic_loss.backward()

            self.critic.optimizer.step()
            self.critic_transformer.optimizer.step()
            self.transformer_optim.step()  # 【论文一致性】Critic 梯度更新 state_transformer

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            alpha = 0
            # 【论文一致性】Actor 使用 detach 后的 state，防止 Actor 梯度传播到 state_transformer
            q_values_pi = th.cat(self.critic.forward(self.critic_transformer(state_for_actor, temporal_feature_short, temporal_feature_long, additional_feature), actions_pi), dim=1)

            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + alpha * th.abs(th.mean(th.sum(replay_data.actions, dim=-1))-1)
            actor_losses.append(actor_loss.item())


            # Optimize the actor
            self.actor.optimizer.zero_grad()
            self.actor_transformer.optimizer.zero_grad()
            actor_loss.backward()

            self.actor.optimizer.step()
            self.actor_transformer.optimizer.step()

            # 【论文一致性】MAE reconstruction loss 仅用于监控，不再单独更新 state_transformer
            # 因为 state_transformer 已经通过 Critic 梯度进行联合训练（论文 Section 4.2）
            transformerloss = (loss_s + loss_ns)/2
            transformer_losses.append(transformerloss.item())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/transformer_loss", np.mean(transformer_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        model_save_path: Optional[str] = None,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
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
                obs_tensor, temporal_short, temporal_long, additional_feature = self._state_transfer_predict(obs)
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

    def _state_transfer_predict(self, x):

        batch_enc1 = x[:, :, :self.in_feat] # [cov+technical_list]

        enc_out, _, output = self.state_transformer(batch_enc1, batch_enc1)

        hidden_channel = enc_out.shape[-1]

        temporal_feature_short = x[:, :, self.in_feat: hidden_channel+self.in_feat]
        temporal_feature_long = x[:, :, hidden_channel+self.in_feat: hidden_channel*2+self.in_feat]
        # temporal_features = th.cat((temporal_feature_short, temporal_feature_long), dim=1)

        additional_feature = x[:, :, hidden_channel*2+self.in_feat:]
        return enc_out, temporal_feature_short, temporal_feature_long, additional_feature


    def _state_transfer(self, x, seed=None, mask_mode='nope'):
        """
        状态转换方法，使用 MAE Transformer 进行状态编码
        为每个模式应用其最快速的实现

        :param x: 输入状态 [bs, stock_num, features]
        :param seed: 可选的随机种子，用于生成 mask。如果提供，确保相同的 seed 生成相同的 mask
        :param mask_mode: mask 模式，可选值：
            - 'stock': 屏蔽股票（默认），随机选择部分股票，屏蔽其全部特征
            - 'feature': 屏蔽技术指标，随机选择部分特征，对所有股票屏蔽这些特征
            - 'mixed': 混合模式，同时随机屏蔽部分股票和部分特征
            - 'nope': 不进行任何屏蔽，保留所有特征和股票
        :return: 编码后的状态、时序特征、附加特征、重建损失
        """
        bs, stock_num = x.shape[0], x.shape[1]
        feat_dim = self.in_feat  # 特征维度 (96)

        batch_enc1 = x[:, :, :feat_dim]  # [bs, stock_num, feat_dim] 包含 cov+technical_list

        if mask_mode == 'stock':
            # ==================== 模式1: 屏蔽股票 - 最快版 ====================
            # 随机选择部分股票，屏蔽其全部特征
            mask = th.ones_like(batch_enc1)

            num_mask = max(1, int(stock_num * 0.01))
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
            # 重建结果output只用于评估重建损失( Shape: [Batch_size, Stock_num, c_out_construction])
            # 中间层特征enc_out真正用于actor和critic(shape: [Batch_size, Stock_num, d_model])
            enc_out, _, output = self.state_transformer(enc_inp, enc_inp)

            # 计算被屏蔽股票的重建损失
            # mask_stock_indices: [bs, num_mask] -> 扩展为 [bs, num_mask, feat_dim]
            gather_idx = mask_stock_indices.unsqueeze(2).expand(-1, -1, feat_dim)  # [bs, num_mask, feat_dim]

            pred = th.gather(output, 1, gather_idx)  # [bs, num_mask, feat_dim]
            true = th.gather(batch_enc1, 1, gather_idx)  # [bs, num_mask, feat_dim]

        elif mask_mode == 'feature':
            # ==================== 模式2: 屏蔽技术指标 - 最快版 ====================
            # 随机选择部分特征，对所有股票屏蔽这些特征
            num_mask = max(1, int(feat_dim * 0.01))
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
            enc_out, _, output = self.state_transformer(enc_inp, enc_inp)

            # 计算被屏蔽特征的重建损失
            # mask_feat_indices: [bs, num_mask] -> 扩展为 [bs, stock_num, num_mask]
            gather_idx = mask_feat_indices.unsqueeze(1).expand(-1, stock_num, -1)  # [bs, stock_num, num_mask]

            pred = th.gather(output, 2, gather_idx)  # [bs, stock_num, num_mask]
            true = th.gather(batch_enc1, 2, gather_idx)  # [bs, stock_num, num_mask]

        elif mask_mode == 'mixed':
            # ==================== 模式3: 混合模式，同时屏蔽股票和特征  - 优化版 ====================
            # 优化版本：减少重复计算，提高性能
            num_stock_mask = max(1, int(stock_num * 0.01))
            num_feat_mask = max(1, int(feat_dim * 0.01))
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

        else:  # if mask_mode == 'nope':
            # ==================== 模式4: 不屏蔽任何特征或股票 ====================
            # 直接使用完整的输入数据，不进行任何掩码操作
            enc_inp = batch_enc1
            enc_out, _, output = self.state_transformer(enc_inp, enc_inp)

            # 由于没有进行掩码，无法计算重建损失，返回零损失
            loss = th.tensor(0.0, device=x.device)

            # 提前返回，跳过其他模式的处理
            hidden_channel = enc_out.shape[-1]
            temporal_feature_short = x[:, :, feat_dim: hidden_channel+feat_dim]
            temporal_feature_long = x[:, :, hidden_channel+feat_dim: hidden_channel*2+feat_dim]

            additional_feature = x[:, :, hidden_channel*2+feat_dim:]
            return enc_out, temporal_feature_short, temporal_feature_long, additional_feature, loss

        loss = self.transformer_criteria(pred, true)

        hidden_channel = enc_out.shape[-1]
        temporal_feature_short = x[:, :, feat_dim: hidden_channel+feat_dim]
        temporal_feature_long = x[:, :, hidden_channel+feat_dim: hidden_channel*2+feat_dim]

        additional_feature = x[:, :, hidden_channel*2+feat_dim:]
        #各元素维度：[bs, stock_num, d_model]， [bs, stock_num, hidden_channel]， [bs, stock_num, hidden_channel]，
        # [bs, stock_num, x.shape[-1] - feat_dim - hidden_channel*2]， loss (标量)
        return enc_out, temporal_feature_short, temporal_feature_long, additional_feature, loss