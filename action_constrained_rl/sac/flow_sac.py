import numpy as np
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.sac.policies import SACPolicy
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import polyak_update
import gym
from action_constrained_rl.nn.additional_layers.flow_layer import FlowLayer


from typing import Any, Dict, List, Optional, Tuple, Type, Union
from random import sample
from stable_baselines3 import SAC
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution


class FlowSACPre(SAC):
    """Add Flow, only use flow as a layer in env, Pre-flow action is used for Learning"""
    def __init__(self, *args,flow_base_gaussian: str, l2_coef:float, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_base_gaussian = flow_base_gaussian
        self.l2_coef = l2_coef

        if self.flow_base_gaussian:
            # If flow is the base distribution we avoid using the tanh
            action_dim = get_action_dim(self.actor.action_space)
            self.actor.action_dist = DiagGaussianDistribution(action_dim)
        self.infeasible_count = 0
        self.flow_update_count = 0
        self.infeasible_states = [] 
    
    def _sample_action_original(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            # Important: Use super predict here. for sample action
            unscaled_action, _ = super().predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action += action_noise()
                if not self.flow_base_gaussian:
                    scaled_action = np.clip(scaled_action, -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _sample_action(self, learning_starts: int, action_noise = None, n_envs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        action, buffer_action = self._sample_action_original(learning_starts, action_noise, n_envs)
        a, s = th.tensor(action, device=FlowLayer.device_id).double(), th.tensor(self._last_obs, device=FlowLayer.device_id).double()
        action_tensor = FlowLayer.flow_forward(a, s).clip(-1, 1)
        action = action_tensor.cpu().numpy()
        
        self.logger.record("const/infeasible_count", self.infeasible_count) 
        self.logger.record("const/infeasible_count_state_list", len(self.infeasible_states)) 
        self.logger.record("train/flow_update_count", self.flow_update_count)
        return action, buffer_action
    
    def predict(self, observation, state = None, episode_start = None, deterministic: bool = False):
        action, _ = super().predict(observation, state, episode_start, deterministic)
        action = FlowLayer.flow_forward(th.tensor(action, device=FlowLayer.device_id).double(), th.tensor(observation, device=FlowLayer.device_id).double()).cpu().numpy()
        return action, _
            

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            if self.flow_base_gaussian:
                ### Add entropy term for gaussian
                norm = th.norm(actions_pi, dim=1).unsqueeze(1)**2
                log_prob += norm*self.l2_coef

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

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                if self.flow_base_gaussian:
                    norm = th.norm(next_actions, dim=1)**2
                    # print("target:", next_actions.shape, norm.shape, next_log_prob.shape)
                    # print(next_actions[:10])
                    next_log_prob += norm*self.l2_coef

                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

