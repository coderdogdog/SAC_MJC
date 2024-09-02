import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from ReplayBuffer import device


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_dim):
        super(Actor, self).__init__()
        self.a_net = nn.Sequential(nn.Linear(state_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim),
                                   nn.ReLU())

        # self.a_net = nn.Sequential(nn.Linear(state_dim, hid_dim),
        #                            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
        #                            nn.Linear(hid_dim, 64), nn.ReLU())

        self.mu_layer = nn.Linear(hid_dim, action_dim)
        self.log_std_layer = nn.Linear(hid_dim, action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)      # 这里clamp 限制更新范围
        # 输出 均值 对数标准差
        return mu, log_std

    def sample_a(self, state, deterministic=False, with_logp=True, reparameterize=True):
        """deterministic 代表是否是确定性策略"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        act_normal = Normal(mean, std)
        if deterministic:
            u = mean
        else:
            # 随机采样
            if reparameterize:
                epsilon = torch.randn_like(std)
                u = mean + std * epsilon
            else:
                u = act_normal.rsample()

        action = torch.tanh(u)

        if with_logp:
            logp_pi_a = (act_normal.log_prob(u).sum(1, keepdim=True)
                         - (2 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2 * u))).sum(1, keepdim=True))
        else:
            logp_pi_a = None

        return action, logp_pi_a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_dim):
        super(Q_Critic, self).__init__()

        self.q_net1 = nn.Sequential(nn.Linear(state_dim + action_dim, hid_dim), nn.ReLU(),
                                    nn.Linear(hid_dim, hid_dim), nn.ReLU(),
                                    nn.Linear(hid_dim, 1), nn.Identity())
        self.q_net2 = nn.Sequential(nn.Linear(state_dim + action_dim, hid_dim), nn.ReLU(),
                                    nn.Linear(hid_dim, hid_dim), nn.ReLU(),
                                    nn.Linear(hid_dim, 1), nn.Identity())

        # self.q_net1 = nn.Sequential(nn.Linear(state_dim + action_dim, hid_dim),
        #                             nn.Linear(hid_dim, hid_dim), nn.ReLU(),
        #                             nn.Linear(hid_dim, 64), nn.ReLU(),
        #                             nn.Linear(64, 1), nn.Identity())
        #
        # self.q_net2 = nn.Sequential(nn.Linear(state_dim + action_dim, hid_dim),
        #                             nn.Linear(hid_dim, hid_dim), nn.ReLU(),
        #                             nn.Linear(hid_dim, 64), nn.ReLU(),
        #                             nn.Linear(64, 1), nn.Identity())

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q_net1(sa)
        q2 = self.q_net2(sa)
        return q1, q2


class SAC_Agent(object):
    def __init__(
            self,
            state_dim=11,
            action_dim=3,
            gamma=0.99,
            hid_dim=128,
            a_lr=3e-4,
            c_lr=3e-4,
            batch_size=256,
            alpha=0.2,
            adaptive_alpha=True
    ):
        self.actor = Actor(state_dim, action_dim, hid_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        # 这个Q网络用于评估Q值与熵
        self.q_critic = Q_Critic(state_dim, action_dim, hid_dim).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)

        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        # self.action_dim = action_dim
        self.gamma = gamma
        self.tau = 0.005    # 滑动更新
        self.batch_size = batch_size

        self.alpha = alpha   # 0.2
        self.adaptive_alpha = adaptive_alpha  # True
        # learned temperature  若是False fixed temperature
        if adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-action_dim, dtype=torch.float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)

    def select_action(self, state, deterministic, with_logprob=False):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a, _ = self.actor.sample_a(state, deterministic, with_logprob)
            # a, _ = self.actor(state, deterministic, with_logprob)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):

        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor.sample_a(s_prime)
            # 原来的Q网络
            target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - dead_mask) * self.gamma * (
                    target_Q - self.alpha * log_pi_a_prime)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        # 原来的
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # self.flag_update_actor = self.flag_update_actor + 1

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks , so you don't waste computational effort
        # computing gradients for them during the policy learning step.

        for params in self.q_critic.parameters():
            params.requires_grad = False

        # 更新Actor网络
        a, log_pi_a = self.actor.sample_a(s)

        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)
        a_loss = (self.alpha * log_pi_a - Q).mean()   # 随机策略

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        # ------------------------ 梯度裁剪 --------------------------------------------

        # grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        #
        # if grad_norm > 0.5:
        #     # Calculate scaling factor
        #     scale = 0.5 / grad_norm
        #     # Scale gradients
        #     for param in self.actor.parameters():
        #         if param.grad is not None:
        #             param.grad.data *= scale
        # --------------------------- 梯度裁剪 --------------------------------------------
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        # 更新阿尔法
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#

        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, episode):
        torch.save(self.actor.state_dict(), "./model/sac_actor{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), "./model/sac_q_critic{}.pth".format(episode))

    def load(self, episode):
        self.actor.load_state_dict(torch.load("./model/sac_actor{}.pth".format(episode)))
        self.q_critic.load_state_dict(torch.load("./model/sac_q_critic{}.pth".format(episode)))
