import copy
import cliff_walking_env


class ValueIteration:
    """价值迭代算法"""
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * self.env.nrow * self.env.ncol  # 初始化状态价值
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
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
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
        print(f"价值迭代进行了{cnt}次后完成")
        self.get_policy()

    def get_policy(self):
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
        return self.pi



env = cliff_walking_env.CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
