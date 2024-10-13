from __future__ import annotations
import matplotlib.pyplot as plt
import gymnasium as gym
from algorithms.monte_carlo_exploring_start import MonteCarloES
from algorithms.monte_carlo_off_policy import MonteCarloOffPolicy
from src.util import create_grids, create_plots

n_episodes = 10000
env = gym.make("Blackjack-v1", sab=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
agent = MonteCarloOffPolicy(env=env)
agent.train(num_episodes=n_episodes)
agent.save_policy()


value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()

value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()


