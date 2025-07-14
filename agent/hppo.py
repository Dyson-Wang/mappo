import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from .memory import HPPOBuffer
from .model import Actor, Critic


class HPPO:
    def __init__(self, num_users, num_states, num_channels, lr_a, lr_c, pmax, gamma, lam, repeat_time, batch_size,
                 eps_clip, w_entropy):
        self.num_state_single = int(num_states / num_users)
        self.actors = [Actor(self.num_state_single, 49, num_channels, pmax).cuda() for _ in range(num_users)]
        self.critics = [Critic(num_states).cuda() for _ in range(num_users)]  # 每个智能体一个critic

        self.optimizer_a = torch.optim.Adam([{'params': actor.parameters(), 'lr': lr_a} for actor in self.actors])
        self.optimizer_c = [torch.optim.Adam(critic.parameters(), lr_c) for critic in self.critics]  # 每个critic一个optimizer

        self.buffer = HPPOBuffer(num_users)

        self.pmax = pmax
        self.gamma = gamma
        self.lam = lam
        self.repeat_time = repeat_time
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.w_entropy = w_entropy

    def select_action(self, state, test=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).cuda()
            actions = []
            for i, actor in enumerate(self.actors):
                prob_points, prob_channels, (power_mu, power_sigma) = actor(state[i*self.num_state_single : i*self.num_state_single+self.num_state_single])

                # point
                dist_point = Categorical(prob_points)
                point = dist_point.sample()
                # channel
                dist_channel = Categorical(prob_channels)
                channel = dist_channel.sample()
                # power
                dist_power = Normal(power_mu, power_sigma)
                power = dist_power.sample()

                if not test:
                    self.buffer.actor_buffer[i].actions_point.append(point)
                    self.buffer.actor_buffer[i].actions_channel.append(channel)
                    self.buffer.actor_buffer[i].actions_power.append(power)

                    self.buffer.actor_buffer[i].logprobs_point.append(dist_point.log_prob(point))
                    self.buffer.actor_buffer[i].logprobs_channel.append(dist_channel.log_prob(channel))
                    self.buffer.actor_buffer[i].logprobs_power.append(dist_power.log_prob(power))

                actions.append((point.item(), channel.item(), power.clamp(1e-10, self.pmax).item()))

                self.buffer.actor_buffer[i].info['prob_points'] = prob_points.tolist()
                self.buffer.actor_buffer[i].info['prob_channels'] = prob_channels.tolist()
                self.buffer.actor_buffer[i].info['power_mu'] = power_mu.item()
                self.buffer.actor_buffer[i].info['power_sigma'] = power_sigma.item()

        return actions

    def update(self):
        self.buffer.build_tensor()
        # 为每个智能体分别生成gae
        with torch.no_grad():
            pred_values_buffer = []
            for i in range(len(self.actors)):
                # 使用全局状态作为critic输入
                states = self.buffer.states_tensor
                pred_values_buffer.append(self.critics[i](states).squeeze())
        target_values_buffer, advantages_buffer = [], []
        for i in range(len(self.actors)):
            tv, adv = self.get_gae(pred_values_buffer[i], i)
            target_values_buffer.append(tv)
            advantages_buffer.append(adv)

        for _ in range(int(self.repeat_time * (len(self.buffer) / self.batch_size))):
            indices = torch.randint(len(self.buffer), size=(self.batch_size,), requires_grad=False).cuda()
            state = self.buffer.states_tensor[indices]
            # ...existing code...
            loss_point = []
            loss_channel = []
            loss_power = []
            for i, actor in enumerate(self.actors):
                logprobs_point = self.buffer.actor_buffer[i].logprobs_point_tensor[indices]
                logprobs_channel = self.buffer.actor_buffer[i].logprobs_channel_tensor[indices]
                logprobs_power = self.buffer.actor_buffer[i].logprobs_power_tensor[indices]

                new_logprobs_point, new_logprobs_channel, new_logprobs_power, \
                entropy_point, entropy_channel, entropy_power = self.eval(i, state[:, i*self.num_state_single : i*self.num_state_single+self.num_state_single], indices)
                self.buffer.actor_buffer[i].info['entropy_point'] = entropy_point.mean().item()
                self.buffer.actor_buffer[i].info['entropy_channel'] = entropy_channel.mean().item()
                self.buffer.actor_buffer[i].info['entropy_power'] = entropy_power.mean().item()

                # point
                ratio_point = (new_logprobs_point - logprobs_point).exp()
                surr1_point = advantages_buffer[i][indices] * ratio_point
                surr2_point = advantages_buffer[i][indices] * torch.clamp(ratio_point, 1 - self.eps_clip, 1 + self.eps_clip)
                loss_point.append((-torch.min(surr1_point, surr2_point) - self.w_entropy * entropy_point).mean())
                # channel
                ratio_channel = (new_logprobs_channel - logprobs_channel).exp()
                surr1_channel = advantages_buffer[i][indices] * ratio_channel
                surr2_channel = advantages_buffer[i][indices] * torch.clamp(ratio_channel, 1 - self.eps_clip, 1 + self.eps_clip)
                loss_channel.append(
                    (-torch.min(surr1_channel, surr2_channel) - self.w_entropy * entropy_channel).mean())
                # power
                ratio_power = (new_logprobs_power - logprobs_power).exp()
                surr1_power = advantages_buffer[i][indices] * ratio_power
                surr2_power = advantages_buffer[i][indices] * torch.clamp(ratio_power, 1 - self.eps_clip, 1 + self.eps_clip)
                loss_power.append((-torch.min(surr1_power, surr2_power) - self.w_entropy * entropy_power).mean())
            loss_a = torch.stack(loss_point + loss_channel + loss_power).mean()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()

            # 分别更新每个critic
            for i in range(len(self.actors)):
                # 使用全局状态作为critic输入
                states = state
                pred_values = self.critics[i](states).squeeze()
                self.optimizer_c[i].zero_grad()
                loss_c = F.mse_loss(pred_values, target_values_buffer[i][indices])
                loss_c.backward()
                self.optimizer_c[i].step()
                self.buffer.info['loss_value'] = loss_c.item()

        self.buffer.init()

    def eval(self, idx, state, indices):
        actions_point = self.buffer.actor_buffer[idx].actions_point_tensor[indices]
        actions_channel = self.buffer.actor_buffer[idx].actions_channel_tensor[indices]
        actions_power = self.buffer.actor_buffer[idx].actions_power_tensor[indices]

        prob_points, prob_channels, (power_mu, power_sigma) = self.actors[idx](state)

        dist_point = Categorical(prob_points)
        dist_channel = Categorical(prob_channels)
        dist_power = Normal(power_mu, power_sigma)

        logprobs_point = dist_point.log_prob(actions_point)
        logprobs_channel = dist_channel.log_prob(actions_channel)
        logprobs_power = dist_power.log_prob(actions_power)

        entropy_point = dist_point.entropy()
        entropy_channel = dist_channel.entropy()
        entropy_power = dist_power.entropy()

        return logprobs_point, logprobs_channel, logprobs_power, entropy_point, entropy_channel, entropy_power

    def get_gae(self, pred_values_buffer, idx):
        with torch.no_grad():
            target_values_buffer = torch.empty(len(self.buffer), dtype=torch.float32).cuda()
            advantages_buffer = torch.empty(len(self.buffer), dtype=torch.float32).cuda()

            next_value = 0
            next_advantage = 0
            for i in reversed(range(len(self.buffer))):
                reward = self.buffer.rewards_tensor[i][idx]
                mask = 1 - self.buffer.is_terminals_tensor[i]
                # value
                target_values_buffer[i] = reward + mask * self.gamma * next_value
                # GAE
                delta = reward + mask * self.gamma * next_value - pred_values_buffer[i]
                advantages_buffer[i] = delta + mask * self.lam * next_advantage

                next_value = pred_values_buffer[i]
                next_advantage = advantages_buffer[i]
            advantages_buffer = (advantages_buffer - advantages_buffer.mean()) / (advantages_buffer.std() + 1e-5)
        return target_values_buffer, advantages_buffer

    def save_model(self, filename, args, info=None):
        dic = {'actor': [actor.state_dict() for actor in self.actors],
               'critic': [critic.state_dict() for critic in self.critics],
               'args': args,
               'info': info}
        torch.save(dic, filename)

    def load_model(self, actor_dicts, critic_dicts):
        for actor, dict in zip(self.actors, actor_dicts):
            actor.load_state_dict(dict)
        for critic, dict in zip(self.critics, critic_dicts):
            critic.load_state_dict(dict)
