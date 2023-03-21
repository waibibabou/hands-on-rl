import random
import time

from chapter5.cliff_walking_env import CliffWalkingEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Dyna_Q:
    """Dyna-Q算法"""
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_planning,n_action=4):
        self.Q_table=np.zeros([ncol*nrow,n_action])#初始化Q表
        self.n_action=n_action
        self.nrow=nrow
        self.ncol=ncol
        self.epsilon=epsilon
        self.gamma=gamma
        self.alpha=alpha
        self.n_planning=n_planning
        self.model=dict()
    def take_action(self,state):
        """用epsilon-greedy选取动作"""
        if np.random.rand()<self.epsilon:
            action=np.random.randint(0,self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action

    def q_learning(self,s0,a0,r,s1):
        """更新Q表"""
        td_error=r+self.gamma*max(self.Q_table[s1])-self.Q_table[s0][a0]
        self.Q_table[s0][a0]+=self.alpha*td_error

    def update(self,s0,a0,r,s1):
        self.q_learning(s0,a0,r,s1)
        self.model[(s0,a0)]=(r,s1)
        for _ in range(self.n_planning):
            (s,a),(r,s_)=random.choice(list(self.model.items()))
            self.q_learning(s,a,r,s_)


    def best_action(self,state):
        """用于最终打印策略"""
        Q_max=max(self.Q_table[state])
        a=[0 for i in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i]==Q_max:
                a[i]=1
        return a



def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    gamma = 0.9
    epsilon = 0.01
    alpha = 0.1
    np.random.seed(0)
    env = CliffWalkingEnv(ncol, nrow)
    agent = Dyna_Q(ncol, nrow, epsilon, alpha, gamma,n_planning)
    num_episodes = 300

    return_list = []  # 用于记录每个episode的回报
    for i in range(10):  # 共显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration-{i}") as pbar:
            for i_episode in range(num_episodes // 10):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({"episode": (num_episodes // 10 * i + i_episode + 1),
                                      "return": f"{np.mean(return_list[-10:])}"})
                pbar.update(1)
    return return_list


np.random.seed(0)
random.seed(0)
n_planning_list=[0,2,20]
for n_planning in n_planning_list:
    print(f"Q-planning的步数为:{n_planning}")
    time.sleep(0.5)
    return_list=DynaQ_CliffWalking(n_planning)
    plt.plot(list(range(len(return_list))),return_list,label=f"Q-planning {n_planning}")

plt.title("Dyna-Q on Cliff Walking")
plt.xlabel('episodes')
plt.ylabel('returns')
plt.legend()
plt.show()



