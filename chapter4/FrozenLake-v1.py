import gym
import Policy_Iteration
env=gym.make('FrozenLake-v1')#创建环境
env=env.unwrapped#解封装才能访问状态转移矩阵P
env.render()#环境渲染

holes=set()
ends=set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2]==1:ends.add(s_[1])
            if s_[3]==True:holes.add(s_[1])
holes=holes-ends
print(f"冰洞的索引为{holes}")
print(f"终点的索引为{ends}")
for a in env.P[14]:
    print(env.P[14][a])

action_meaning=['<','v','>','^']
theta=1e-5
gamma=0.9
agent=Policy_Iteration.PolicyIteration(env,theta,gamma)
agent.policy_iteration()
Policy_Iteration.print_agent(agent,action_meaning,[5,7,11,12],[15])

