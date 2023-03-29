import torch
from torch.nn import Linear
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch.nn.functional as F

import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1=Linear(state_dim,hidden_dim)
        self.fc2=Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)

class REINFORCE:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,device):
        self.device=device
        self.gamma=gamma
        self.policy_net=PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.policy_net.parameters(),lr=learning_rate)

    def take_action(self,state):
        """根据动作概率分布随机采样"""
        state=torch.tensor(np.array([state]),dtype=torch.float).to(self.device)
        probs=self.policy_net(state)
        action_dist=torch.distributions.Categorical(probs)
        action=action_dist.sample()
        return action.item()

    def update(self,transition_dict):
        reward_list=transition_dict['rewards']
        state_list=transition_dict['states']
        action_list=transition_dict['actions']
        self.optimizer.zero_grad()
        G=0
        for i in range(len(reward_list)-1,-1,-1):
            reward=reward_list[i]
            #这里相当于batch_size为1，也要将输入的第一个维度变为batch_size
            state=torch.tensor(np.array([state_list[i]]),dtype=torch.float).to(self.device)
            action=torch.tensor(np.array([action_list[i]]),dtype=torch.int64).view(-1,1).to(self.device)
            log_prob=torch.log(self.policy_net(state).gather(1,action))
            G=self.gamma*G+reward
            loss=-G*log_prob
            loss.backward()
        self.optimizer.step()


learning_rate=1e-3
num_episodes=1000
hidden_dim=128
gamma=0.98
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name='CartPole-v1'
env=gym.make(env_name)
env.reset(seed=0)
torch.manual_seed(0)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n
agent=REINFORCE(state_dim,hidden_dim,action_dim,learning_rate,gamma,device)

return_list=[]
for i in range(10):
    with tqdm(total=num_episodes//10,desc=f'Iteration-{i}') as pbar:
        for i_episode in range(num_episodes//10):
            state=env.reset()
            episode_return=0
            done=False
            transition_dict={'states':[],'actions':[],'next_states':[],'rewards':[],'dones':[]}
            while not done:
                action=agent.take_action(state)
                next_state,reward,done,info=env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state=next_state
                episode_return+=reward
            return_list.append(episode_return)
            agent.update(transition_dict)

            pbar.set_postfix({'return':f'{np.mean(return_list):.3f}','episode':f'{num_episodes//10*i +i_episode+1}'})
            pbar.update()


return_list=rl_utils.moving_average(return_list,9)
plt.plot(list(range(len(return_list))),return_list)
plt.title(f'REINFORCE on {env_name}')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.axhline(200)
plt.show()








