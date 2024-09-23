import numpy as np
import time

# from bandit import BernoulliBandit

class Policy:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = [0] * self.bandit.n
        self.actions = []
        self.rewards = []
        self.cumulative_regret = []

    def step(self):
        raise NotImplementedError

    def run(self, N):
        for _ in range(N):
            (i,r) = self.step()
            self.counts[i] += 1
            self.actions.append(i)
            self.rewards.append(r)
            regret = self.bandit.top_proba - self.bandit.arms[i]
            if len(self.cumulative_regret) != 0:
                regret += self.cumulative_regret[-1]
            self.cumulative_regret.append(regret)


class EpsilonGreedy(Policy):
    def __init__(self, bandit, epsilon, init_proba=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = [init_proba] * self.bandit.n

    def step(self):
        if np.random.random() < self.epsilon:
            i = np.random.randint(0, self.bandit.n)
        else:
            i = self.estimates.index(max(self.estimates))
        reward = self.bandit.get_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (reward - self.estimates[i])
        return (i, reward)


class UCB1(Policy):
    def __init__(self, bandit, init_proba=1.0):
        super().__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n

    def step(self):
        self.t += 1
        i = max(
            range(self.bandit.n),
            key=lambda x: self.estimates[x] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[x]))
        )
        reward = self.bandit.get_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (reward - self.estimates[i])
        return (i, reward)


class ThompsonSampling(Policy):
    def __init__(self, bandit, init_a=1, init_b=1):
        super().__init__(bandit)
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    def step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        reward = self.bandit.get_reward(i)
        self._as[i] += reward
        self._bs[i] += (1 - reward)

        return (i, reward)