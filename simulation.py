from bandit import BernoulliBandit
from policies import EpsilonGreedy, UCB1, ThompsonSampling
from plots import plot_regret, plot_arm_selection

def experiment(K, N):
    bandit = BernoulliBandit(K, probas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    policies = [
        EpsilonGreedy(bandit, epsilon=0.01, init_proba=1.),
        UCB1(bandit, init_proba=1.0),
        ThompsonSampling(bandit, 1, 1)
    ]
    names = [
        r'$\epsilon$' + '-Greedy',
        'UCB1',
        'Thompson Sampling'
    ]

    for policy in policies:
        policy.run(N)

    plot_regret(policies, names)
    plot_arm_selection(policies, names)



if __name__ == '__main__':
    experiment(10, 10000)