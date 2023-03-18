import copy
import cliff_walking_env


class PolicyIteration:
    """
    动态规划中的策略迭代算法
    适用范围有限，需要知道P与奖励函数，并且状态与动作空间离散且有限
    """

    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * self.env.nrow * self.env.ncol  # 初始化状态价值
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]  # 初始化为均匀随机策略

    def policy_evaluation(self):
        """策略评估"""
        cnt = 0  # 用于记录迭代了多少次
        while 1:
            cnt += 1
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.nrow * self.env.ncol):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.v[next_state] * (1 - done) * self.gamma)
                    qsa_list.append(qsa * self.pi[s][a])
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
        print(f"策略评估进行了{cnt}次后完成")

    def policy_improvement(self):
        """策略提升"""
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if new_pi == old_pi: break


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s'%('%.3f'%agent.v[i*agent.env.ncol+j]), end=' ')
        print()
    print("策略:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i*agent.env.ncol+j) in disaster:
                print('****',end=' ')
            elif (i*agent.env.ncol+j) in end:
                print('EEEE',end=' ')
            else:
                a=agent.pi[i*agent.env.ncol+j]
                temp=''
                for k in range(len(action_meaning)):
                    if a[k]>0:
                        temp+=action_meaning[k]
                    else:
                        temp+='o'

                print(temp,end=' ')
        print()


env = cliff_walking_env.CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent,action_meaning,list(range(37,47)),[47])