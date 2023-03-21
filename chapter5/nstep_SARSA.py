import matplotlib.pyplot as plt
import numpy as np
import gym
import cliff_walking_env
from tqdm import tqdm

class nstep_Sarsa:
    """多步SARSA算法"""
    def __init__(self,n,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        self.Q_table=np.zeros([ncol*nrow,n_action])#初始化Q表
        self.n_action=n_action
        self.nrow=nrow
        self.ncol=ncol
        self.epsilon=epsilon
        self.gamma=gamma
        self.alpha=alpha
        self.n=n
        self.state_list=[]#保存之前的状态
        self.action_list=[]#保存之前的动作
        self.reward_list=[]#保存之前的奖励

    def take_action(self,state):
        """用epsilon-greedy选取动作"""
        if np.random.rand()<self.epsilon:
            action=np.random.randint(0,self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action

    def update(self,s0,a0,r,s1,a1,done):
        """更新Q表"""
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list)==self.n:
            G=self.Q_table[s1][a1]
            for i in reversed(range(self.n)):
                G=self.gamma*G+self.reward_list[i]

                if done and i>0:
                    s=self.state_list[i]
                    a=self.action_list[i]
                    self.Q_table[s,a]+=self.alpha*(G-self.Q_table[s,a])
            s=self.state_list.pop(0)
            a=self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s,a]+=self.alpha*(G-self.Q_table[s,a])
        if done:
            self.action_list=[]
            self.reward_list=[]
            self.state_list=[]


    def best_action(self,state):
        """用于最终打印策略"""
        Q_max=max(self.Q_table[state])
        a=[0 for i in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i]==Q_max:
                a[i]=1
        return a

ncol=12
nrow=4
gamma=0.9
epsilon=0.1
alpha=0.1
n_step=5
np.random.seed(0)
env=cliff_walking_env.CliffWalkingEnv(ncol,nrow)
agent=nstep_Sarsa(n_step,ncol,nrow,epsilon,alpha,gamma)
num_episodes=500

return_list=[]#用于记录每个episode的回报
for i in range(10):#共显示10个进度条
    with tqdm(total=int(num_episodes/10),desc=f"Iteration-{i}") as pbar:
        for i_episode in range(num_episodes//10):
            episode_return=0
            state=env.reset()
            action=agent.take_action(state)
            done=False
            while not done:
                next_state,reward,done=env.step(action)
                next_action=agent.take_action(next_state)
                episode_return+=reward#这里回报的计算不用乘折扣因子
                agent.update(state,action,reward,next_state,next_action,done)
                action=next_action
                state=next_state
            return_list.append(episode_return)
            if (i_episode+1)%10==0:
                pbar.set_postfix({"epsisode":(num_episodes//10*i+i_episode+1),"return":f"{np.mean(return_list[-10:])}"})
            pbar.update(1)


x=list(range(len(return_list)))
plt.plot(x,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on Cliff Walking')
plt.show()


acion_meaning=['^','v','<','>']

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("策略:")
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i*env.ncol+j) in disaster:
                print('****',end=' ')
            elif (i*env.ncol+j) in end:
                print('EEEE',end=' ')
            else:
                a=agent.best_action(i*env.ncol+j)
                temp=''
                for k in range(len(action_meaning)):
                    if a[k]>0:
                        temp+=action_meaning[k]
                    else:
                        temp+='o'

                print(temp,end=' ')
        print()
print_agent(agent,acion_meaning,list(range(37,47)),[47])



