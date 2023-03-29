import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
import rl_utils
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt


class TwoLayerFC(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation=F.relu, out_fn=lambda x: x):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, output_dim)

        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class DDPG:
    def __init__(self, input_dim_actor, output_dim_actor, input_dim_critic, hidden_dim, discrete, action_bound, sigma,
                 actor_lr, critic_lr, tau, gamma, device):
        out_fn = (lambda x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(input_dim_actor, output_dim_actor, hidden_dim, activation=F.relu, out_fn=out_fn).to(
            device)
        self.target_actor = TwoLayerFC(input_dim_actor, output_dim_actor, hidden_dim, activation=F.relu,
                                       out_fn=out_fn).to(device)

        self.critic = TwoLayerFC(input_dim_critic, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(input_dim_critic, 1, hidden_dim).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.device = device
        self.tau = tau
        self.sigma = sigma
        self.action_dim = output_dim_actor
        self.action_bound = action_bound

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        temp=self.actor(state)
        action = temp.item()
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(torch.cat([next_states, self.target_actor(next_states)], dim=1)) #torch.cat操作不会升维
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(torch.cat([states, actions], dim=1)), q_targets.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(torch.cat([states, self.actor(states)], dim=1)))#这里主要要重新过一下actor，如果直接用字典里的action则loss对actor的参数的就没有梯度了
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)


actor_lr = 5e-4
critic_lr = 5e-3
num_episodes = 200

hidden_dim = 64
gamma = 0.98
tau = 0.005
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'Pendulum-v1'
env = gym.make(env_name)
random.seed(0)
torch.manual_seed(0)
env.reset(seed=0)
np.random.seed(0)

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False, action_bound, sigma, actor_lr, critic_lr,
             tau, gamma, device)
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

my_return = rl_utils.moving_average(return_list, 9)
plt.title(f'DDPG on {env_name}')
plt.plot(list(range(len(my_return))), my_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()
