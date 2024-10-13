import random
#import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
from Model.TGCN_me import TGCN

"""
    States:(10)
        current states + predictions(5)
        选择预测未来24h的数据，然后合成为一个24*hidden的，具体看visio
        electricity purchase price[a1, a2, ..., a24]
        PV power[a1, a2, ..., a24]
        electricity demand[a1, a2, ..., a24]
        heating demand[a1, a2, ..., a24]
        cooling demand[a1, a2, ..., a24]

        battery storage level
        hydrogen tank level
        heating water tank level
        cooling water tank level

        t
        [24, 24, 24, 24, 24, 1,1,1,1,1]

    Actions:
        P_g, P_bss, P_H, P_tes, P_AC, P_css, P_ely, P_fc
        a_g, a_bss, a_B, a_tes, a_AC, a_css, a_ely, a_fc
        

"""


def data_extraction(x, time_step):
    e_prices = x[:, :time_step]
    # e_selling = x[:, time_step : time_step*2]
    # H_price  = x[:, time_step*2 : time_step*3]
    pv_power = x[:, time_step : time_step*2]
    Q_ED     = x[:, time_step*2 : time_step*3]
    Q_HD     = x[:, time_step*3 : time_step*4]
    Q_CD     = x[:, time_step*4 : time_step*5]
    tgcn_input = torch.stack((e_prices, pv_power, Q_ED, Q_HD, Q_CD), dim=1)
    x_rest = x[:, -5:]
    return tgcn_input, x_rest

class Actor(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, units, stack_cnt, time_step, horizon, dropout_rate, leaky_rate, action_bound, scaled_action_indices, device):
        super(Actor, self).__init__()
        self.unit = units
        self.stack_cnt = stack_cnt
        self.time_step = time_step  # i.e. K
        # self.multi_layer = multi_layer
        self.horizon = horizon
        self.dropout_rate = dropout_rate
        self.leaky_rate = leaky_rate
        self.device = device
        self.action_bound = action_bound
        self.scaled_action_indices = scaled_action_indices

        self.tgcn = TGCN(self.unit, self.stack_cnt, self.time_step, self.horizon, self.dropout_rate, self.leaky_rate, self.device)
        self.state_dim = state_dim
        self.fc1 = torch.nn.Linear(self.unit * self.horizon + self.state_dim - self.time_step * self.unit, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # e_prices = x[:, :self.time_step]
        # pv_power = x[:, self.time_step+4: self.time_step+4 + self.time_step]
        # Q_ED     = x[:, self.time_step+4 +self.time_step:self.time_step+4 +self.time_step*2]
        # Q_HD     = x[:, self.time_step+4 +self.time_step*2:self.time_step+4 +self.time_step*3]
        # Q_CD     = x[:, self.time_step+4 +self.time_step*3:self.time_step+4 +self.time_step*4]
        
        # tgcn_input = torch.stack((e_prices, pv_power, Q_ED, Q_HD, Q_CD), dim=1)
        # x_rest = torch.cat((x[:, self.time_step:self.time_step+4], x[:, -1].unsqueeze(-1)), dim=1)
        tgcn_input, x_rest = data_extraction(x, self.time_step)
        # 提取时空特征
        bat, _, _ = tgcn_input.size()
        tgcn_output = self.tgcn(tgcn_input).view(bat, -1)    # shape: batch size, 5 * 24
        x = torch.cat((tgcn_output, x_rest), dim=1)
        # 获取actions
        x = F.relu(self.fc1(x))
        a = torch.sigmoid(self.fc2(x)) * self.action_bound
        for idx in self.scaled_action_indices:
            a[:, idx] = a[:, idx] * 2 - 1
        return a
    
class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, units, stack_cnt, time_step, horizon, dropout_rate, leaky_rate, action_bound, scaled_action_indices, device):
        super(Critic, self).__init__()
        self.unit = units
        self.stack_cnt = stack_cnt
        self.time_step = time_step  # 24
        # self.multi_layer = multi_layer
        self.horizon = horizon
        self.dropout_rate = dropout_rate
        self.leaky_rate = leaky_rate
        self.device = device
        self.action_bound = action_bound
        self.scaled_action_indices = scaled_action_indices

        self.tgcn = TGCN(self.unit, self.stack_cnt, self.time_step, self.horizon, self.dropout_rate, self.leaky_rate, self.device)
        self.fc1 = torch.nn.Linear(self.unit * self.horizon + state_dim - self.time_step * self.unit + action_dim, 2 * hidden_dim)
        self.fc2 = torch.nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc3 = torch.nn.Linear(2 * hidden_dim, 1)
    
    def forward(self, x, a):
        """
            x和a如何输入到GCN，只输入x还是x和a都输入？
        """
        tgcn_input, x_rest = data_extraction(x, self.time_step)
        # 提取时空特征
        bat, _, _ = tgcn_input.size()
        tgcn_output = self.tgcn(tgcn_input).view(bat, -1)    # shape: batch size, 5 * 24
        input = torch.cat((tgcn_output, x_rest, a), dim=1)

        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        return output
    

class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, scaled_action_indices, sigma, actor_lr, critic_lr, tau, gamma, units, stack_cnt, time_step, horizon, dropout_rate, leaky_rate, device):
        self.actor = Actor(state_dim, hidden_dim, action_dim, units, stack_cnt, time_step, horizon, dropout_rate, leaky_rate, action_bound, scaled_action_indices, device).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim, units, stack_cnt, time_step, horizon, dropout_rate, leaky_rate, action_bound, scaled_action_indices, device).to(device)

        self.target_actor = Actor(state_dim, hidden_dim, action_dim, units, stack_cnt, time_step, horizon, dropout_rate, leaky_rate, action_bound, scaled_action_indices, device).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim, units, stack_cnt, time_step, horizon, dropout_rate, leaky_rate, action_bound, scaled_action_indices, device).to(device)

        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau      # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.num_update = 0   # 更新次数

    def take_action(self, state, program ):
        state = torch.tensor([state], dtype = torch.float).to( self.device )
        action = self.actor(state).cpu().detach().numpy()
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        self.num_update += 1                            # 更新次数加一
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
     
        ###梯度裁剪
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)

        self.critic_optimizer.step()

        # 每两次更新参数
        if self.num_update % 2 == 0:
            actor_loss = -torch.mean(self.critic(states, self.actor(states)))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            ###梯度裁剪
            for param in self.actor.parameters():
                param.grad.data.clamp_(-1, 1)

            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
            self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
    
    def save_results(self, file_name):
        #torch.save(self.actor.state_dict(), f'result/{file_name}_actor.pth')
        #torch.save(self.critic.state_dict(), f'result/{file_name}_critic.pth')
        model = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(model, f'result/{file_name}')  ###打包存储训练得到的模型

    def load_results(self, file_name):
        #self.actor.load_state_dict(torch.load( f'result/{file_name}_actor.pth'))
        #self.critic.load_state_dict(torch.load(f'result/{file_name}_critic.pth'))

        model = torch.load(f'result/{file_name}')
        self.actor.load_state_dict( model['actor'] )
        self.critic.load_state_dict( model['critic'] )


class PPO:
    def __init__(self):
        pass