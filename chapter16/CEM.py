import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn.functional as F
import random
import collections
import itertools
from scipy.stats import truncnorm


class CEM:
    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound, lower_bound):
        self.n_sequence = n_sequence
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.elite_ratio = elite_ratio
        self.fake_env = fake_env

    def optimize(self, state, init_mean, init_var):
        mean, var = init_mean, init_var
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        state = np.tile(state, (self.n_sequence, 1))

        for _ in range(5):
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            # 生成动作序列
            action_sequences = [X.rvs() for _ in range(self.n_sequence)] * np.sqrt(constrained_var) + mean

            returns = self.fake_env.propagate(state, action_sequences)[:, 0]

            elites = action_sequences[np.argsort(returns)][-int(self.elite_ratio * self.n_sequence):]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            # 更新动作序列分布
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var

        return mean
