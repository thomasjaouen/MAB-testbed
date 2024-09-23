import numpy as np
import time

class BernoulliBandit:
    def __init__(self, n, probas=None):
        assert probas == None or len(probas) == n
        self.n = n
        np.random.seed(int(time.time()))
        if probas:
            self.arms = probas
        else:
            self.arms = [np.random.random() for _ in range(self.n)]
        self.top_proba = max(self.arms)

    def get_reward(self, i):
        return np.random.random() < self.arms[i]