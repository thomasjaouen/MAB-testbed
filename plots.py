import matplotlib.pyplot as plt

def plot_regret(policies, names, output='regret.png'):
    plt.figure(figsize=(12,7))
    for name, policy in zip(names,policies):
        regret = policy.cumulative_regret
        plt.plot(range(1, len(regret)+1), regret, label=name)
    plt.title('Cumulative regret over time')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative regret')
    plt.legend()
    plt.xlim(-1,len(policies[0].cumulative_regret)+1)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()

def plot_arm_selection(policies, names, output='arm_selection.png'):
    plt.figure(figsize=(15, 5))

    for i in range(1, len(policies)+1):
        plt.subplot(1, 3, i)
        name, policy = names[i-1], policies[i-1]
        # 1. Get the proportions
        counts = [0] * policy.bandit.n
        proportions = [[] for _ in range(policy.bandit.n)]
        for i in range(len(policy.actions)):
            action = policy.actions[i]
            counts[action] += 1
            for idx, proportion in enumerate(proportions):
                proportion.append(counts[idx]/(i+1))
        # 2. Plot the proportions
        for idx, proportion in enumerate(proportions):
            LABEL = f'Bandit {idx}'+r' ($\theta$ = ' + f'{policy.bandit.arms[idx]})'
            plt.plot(range(1, len(proportion)+1), proportion, label=LABEL)
        plt.title(f'Proportion of arm selection | {name}')
        plt.xlabel('Step')
        plt.ylabel('Proportion')
        plt.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
