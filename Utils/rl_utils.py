from tqdm import tqdm
import streamlit as st
import numpy as np
import torch
import collections
import random
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    agent.save_results()
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, seed, program, file_name):
    set_seed(seed)
    return_list    = []
    trace_list     = []
    violation_list = []
    cost_list      = []
    if program == 'test':
        state = env.state_normalization(env.state) ###获取环境的初始状态
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:     
            for i_episode in range( int(num_episodes/10) ):
                current_episode = num_episodes/10 * i + i_episode
                episode_return    = 0    ### 计算每个episode的汇报
                episode_violation = 0    ### 计算每个episode的约束违反惩罚
                episode_cost      = 0    ### 计算每个episode累积费用

                trace = {} ### store the epoch trajectories
                trace['day']       = []  ### 存储日期  
                trace['state' ]    = []
                trace['action']    = []
                trace['decision']  = []
                trace['reward']    = []
                trace['violation'] = []
                trace['cost']      = []
                
                if program == 'train': 
                    state = env.reset()            ### 如果为训练模式，则每个episode都是随机选的一天
                elif program == 'test':         
                    agent.load_results(file_name)  ### 如果为测试模式，则加载训练好的网络参数，测试从第0开始，顺序推进

                trace['day'].append(env.day)
                done = False
                while not done:
                    action = agent.take_action(state, program)[0]            ### 获取决策（未加噪声）
                    if program == 'train':
                           
                        k = ( np.log(3) - np.log(0.02) )/(0.4 * num_episodes)
                        sigma = np.maximum( 3 * np.exp(-k * current_episode), 0.02) 
                        action = action + sigma * np.random.randn(env.action_dim)          # 给动作添加噪声，增加探索  

                    decision = env.get_decision(action)                             ### 获取行动   
                    next_state, reward, done, violation, cost = env.step(action)    ### 这里next_state是归一化后的状态
              
                    trace['state'].append(env.state)   # 存储当前状态 (未归一化状态)
                    trace['action'].append(action)     # 存储当前行动
                    trace['decision'].append(decision) # 存储当前决策
                    trace['reward'].append(reward)     # 存储当前收益
                    trace['violation'].append(violation)
                    trace['cost'].append(cost)

                    replay_buffer.add(state, action, reward, next_state, done)   ### 添加经验到经验池
                    state = next_state
                    episode_return += reward
                    episode_violation += violation
                    episode_cost += cost
                    
                    #显示action的探索过程
                    if program == 'train':  ### 如果为训练模式，检查buffer是否存储足够的经验，并判定是否启动训练过程
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                            agent.update(transition_dict)

                return_list.append(episode_return)
                trace_list.append(trace)
                violation_list.append(episode_violation)
                cost_list.append(episode_cost)
                
                if (i_episode + 1) % 10 == 0 or program == 'test':
                    pbar.set_postfix({
                        'Iteration': i,
                        'Episode': f'{i_episode + 1}/{int(num_episodes/10)}',
                        'Return': f'{np.mean(return_list[-100:]):.3f}',
                        'Cost': f'{np.mean(cost_list[-100:]):.3f}',
                        'Violation': f'{np.mean(violation_list[-100:]):.3f}'
                    })

                pbar.update(1)
    if program == 'train':
        agent.save_results(file_name)   ### 保存网络参数
        print('Finished training and Models Saved !')
    elif program == 'test':
        np.savez(f'result/{file_name[:-4]}_traces.npz', trace_list = trace_list)
        print('Finished testing and Traces Saved !' )
    return return_list, trace_list, violation_list, cost_list






def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                