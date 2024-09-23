# Multi-Armed (Bernoulli) Bandit Strategies Comparison

This repository implements a simulation to compare different strategies for solving the Multi-Armed Bandit problem, specifically for the case of a Bernoulli Bandit. Three classic strategies are included: Epsilon-Greedy, UCB1, and Thompson Sampling. The simulation allows to visualize and compare the cumulative regret and arm selection proportions for each strategy.

## Repo structure

```
.
├── arm_selection.png
├── bandit.py
├── plots.py
├── policies.py
├── regret.png
└── simulation.py
```

- `bandit.py`: Defines the `BernoulliBandit` class, representing a multi-armed bandit where each arm provides a reward based on a Bernoulli distribution with a given probability.
- `policies.py`: Implements three bandit-solving strategies:
  - **Epsilon-Greedy**: Selects a random arm with a probability `epsilon`, and exploits the best-known arm with probability `1 - epsilon`.
  - **UCB1**: Upper Confidence Bound algorithm balances exploration and exploitation by calculating confidence bounds for each arm.
  - **Thompson Sampling**: Bayesian approach to balancing exploration and exploitation using Beta distributions for each arm.
- `plots.py`: Contains plotting functions for visualizing the performance of the strategies, such as cumulative regret and the proportion of arm selections.
- `simulation.py`: Runs the experiment with the bandit and strategies, producing plots that compare the performance of the strategies over time.

## How to Run

1. Clone the repository
```{bash}
git clone https://github.com/thomasjaouen/MAB-testbed.git
cd multi-armed-bandit
```
2. Install the required dependencies
```
pip install matplotlib numpy
```
3. Run the simulation
```
python simulation.py
```
By default, the experiment runs with 10 arms and 10,000 steps. The results include a plot of cumulative regret and the proportion of arm selections for each strategy.

## Example outputs

The simulation outputs two plots:
- `regret.png`: Shows the regret over time for each strategy.
- `arm_selection.png`: Displays the proportion of times each arm is selected throughout the simulation.
