import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """伯努利多臂赌博机，K为拉杆数目"""

    def __init__(self, K):
        self.probs = np.random.uniform(size=(K))  # 随机生成K个0到1随机数
        self.best_idx = np.argmax(self.probs)  # 获取概率最大的拉杆
        self.best_prob = max(self.probs)
        self.K = K

    def step(self, k):
        """执行动作获取奖励"""
        if np.random.rand() < self.probs[k]:
            return 1
        return 0


np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print(f"随机生成了一个有{K}个拉杆的赌博机")
print(f"概率最大的为拉杆{bandit_10_arm.best_idx}，概率为{bandit_10_arm.best_prob:.2f}")


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.regret = 0  # 记录累计懊悔值
        self.regrets = []
        self.actions = []
        self.counts = np.zeros(shape=(self.bandit.K))  # 记录每个拉杆的拉动次数，用于计算回报期望

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.update_regret(k)
            self.actions.append(k)



#epsilon贪婪策略
class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=0.01,init_prob=1):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon=epsilon
        self.estimates=[init_prob]*bandit.K

    def run_one_step(self):
        if np.random.rand()<self.epsilon:
            k=np.random.randint(0,self.bandit.K)
        else:
            k=np.argmax(self.estimates)
        r=self.bandit.step(k)
        self.estimates[k]+=1/(self.counts[k]+1)*(r-self.estimates[k])
        return k


def plot_results(slovers,solver_names):
    for i in range(len(slovers)):
        time_list=list(range(len(slovers[i].regrets)))
        plt.plot(time_list,slovers[i].regrets,label=solver_names[i])
    plt.xlabel("Time steps")
    plt.ylabel("cumulative regrets")
    plt.title(f"{slovers[0].bandit.K}-armed bandit")
    plt.legend()
    plt.show()

np.random.seed(1)
epsilon_greedy_solver=EpsilonGreedy(bandit_10_arm,0.01)
epsilon_greedy_solver.run(5000)
print(f"epsilon贪婪法的累计懊悔为:{epsilon_greedy_solver.regret:.2f}")
plot_results([epsilon_greedy_solver],["Epsilon-greedy"])


np.random.seed(0)
epsilons=[1e-4,0.01,0.1,0.25,0.5]
epsilon_greedy_solver_list=[EpsilonGreedy(bandit_10_arm,i) for i in epsilons]
epsilon_greedy_solver_names=[f"epsilon={i}" for i in epsilons]
for i in range(len(epsilon_greedy_solver_list)):
    epsilon_greedy_solver_list[i].run(5000)
plot_results(epsilon_greedy_solver_list,epsilon_greedy_solver_names)



class DecayingEpsilonGreedy(Solver):
    """尝试epsilon随着时间衰减的算法"""
    def __init__(self,bandit,init_prob=1):
        super(DecayingEpsilonGreedy,self).__init__(bandit)
        self.estimates=[init_prob]*self.bandit.K
        self.tatal_count=0
    def run_one_step(self):
        self.tatal_count+=1
        if np.random.rand()<1/self.tatal_count:
            k=np.random.randint(0,self.bandit.K)
        else:
            k=np.argmax(self.estimates)
        r=self.bandit.step(k)
        self.estimates[k]+=1/(self.counts[k]+1)*(r-self.estimates[k])
        return k
np.random.seed(1)
decaying_epsilon_greedy_solver=DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
plot_results([decaying_epsilon_greedy_solver],["DecayingEpsilonGreedy"])


class UCB(Solver):
    """上置信界算法"""
    def __init__(self,bandit,coef,init_prob=1):
        super(UCB,self).__init__(bandit)
        self.coef=coef
        self.total_count=0
        self.estimates=[init_prob]*self.bandit.K
    def run_one_step(self):
        self.total_count+=1
        ucb=self.estimates+self.coef*np.sqrt(np.log(self.total_count)/(2*self.counts+2))
        k=np.argmax(ucb)
        r=self.bandit.step(k)
        self.estimates[k]+=1/(self.counts[k]+1)*(r-self.estimates[k])
        return k

np.random.seed(1)
coef=1
UCB_solver=UCB(bandit_10_arm,coef)
UCB_solver.run(5000)
plot_results([UCB_solver],["UCB"])


class ThompsonSampling(Solver):
    """汤普森采样算法"""
    def __init__(self,bandit):
        super(ThompsonSampling,self).__init__(bandit)
        self._a=np.ones(self.bandit.K)
        self._b=np.ones(self.bandit.K)
    def run_one_step(self):
        samples=np.random.beta(self._a,self._b)
        k=np.argmax(samples)
        r=self.bandit.step(k)
        self._a[k]+=r
        self._b[k]+=1-r
        return k

np.random.seed(1)
thompson_sampling_solver=ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
plot_results([thompson_sampling_solver],["ThompsonSampling"])
