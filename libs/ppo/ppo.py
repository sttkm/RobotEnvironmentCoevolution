
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.buffers import RolloutBuffer

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class PPO:
    def __init__(self,
        envs,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='cpu',
        lr_decay=True,
        max_iter=None,
        iter=0):

        self.rollout_buffer = RolloutBuffer(
            n_steps,
            envs.observation_space,
            envs.action_space,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_envs=envs.num_envs,
        )
        self.envs = envs
        self.n_steps = n_steps
        self.action_space = envs.action_space
        self.observation_space = envs.observation_space
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.lr_decay = lr_decay
        self.max_iter = max_iter
        self.initial_lr = learning_rate

        self._last_obs = self.envs.reset()
        self._last_episode_starts = np.ones((self.envs.num_envs,), dtype=bool)

        self.iter = iter

    def set_envs(self, envs):
        self.envs = envs

    def update_lr(self, optimizer):
        lr = self.initial_lr * (1 - (self.iter / self.max_iter))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, policy, optimizer):
        if self.lr_decay:
            self.update_lr(optimizer)

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                optimizer.step()

    def collect_rollouts(self, policy, n_rollout_steps):
        n_steps = 0
        self.rollout_buffer.reset()
        while n_steps < n_rollout_steps:

            with torch.no_grad():
                actions, values, log_probs = policy(self._last_obs)
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = self.envs.step(actions)

            n_steps += 1

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                    ):
                    terminal_obs = np.array(infos[idx]["terminal_observation"])
                    terminal_obs = terminal_obs.reshape((-1,) + self.observation_space.shape)
                    with torch.no_grad():
                        terminal_value = policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            self.rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = policy.predict_values(new_obs)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        return

    def step(self, policy, optimizer):
        self.collect_rollouts(policy, n_rollout_steps=self.n_steps)
        self.train(policy, optimizer)
        self.iter += 1
