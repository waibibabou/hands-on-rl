import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear, ReLU, Softmax
import gym
import rl_utils
from tqdm import tqdm


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, action_dim)
        self.ac=ReLU()

    def forward(self, x):
        x = self.ac(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                 dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            return self.q_net(state).argmax().item()

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max(1)[0].item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # 第一个维度为batch_size
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 为了后续能够相加，使用view(-1,1)将一维tensor转为二维
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)

        q_values = self.q_net(states).gather(1, actions).view(batch_size)
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action).view(batch_size)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0]
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # 书上都是转为二维tensor计算的，我这里用的都是一维tensor，结果相同
        dqn_loss = loss_fn(q_values, q_targets)

        self.optimizer.zero_grad()  # pytorch中默认梯度会积累，所以每次更新前将梯度置零
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


loss_fn = torch.nn.MSELoss(reduction='mean')
lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'Pendulum-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 11


def dis_to_con(discrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)


def train_DQN(agent: DQN, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=num_episodes // 10, desc=f'Iteration-{i}') as pbar:
            for i_episode in range(num_episodes // 10):
                episode_return = 0
                state = env.reset()  # 返回一个随机的初始状态
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)
                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, done, info = env.step([action_continuous])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': (num_episodes // 10 * i + i_episode + 1),
                                      'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update()
    return return_list, max_q_value_list


random.seed(0)
np.random.seed(0)
env.reset(seed=0)  # 使得后面env.reset()返回的状态不再随机
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device,'DoubleDQN')
return_list, max_q_value_list = train_DQN(agent, env, num_episodes,replay_buffer,minimal_size,batch_size)
return_list=rl_utils.moving_average(return_list,5)

plt.title(f'DQN on {env_name}')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.plot(list(range(len(return_list))),return_list)
plt.show()


plt.title(f'DQN on {env_name}')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.plot(list(range(len(max_q_value_list))),max_q_value_list)
plt.axhline(0,c='orange',ls='--')
plt.axhline(10,c='red',ls='--')
plt.show()







