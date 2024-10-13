from collections import defaultdict
import random
import gymnasium as gym
from gymnasium import Env
import numpy as np
from tqdm import tqdm

from src.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from src.algorithms.trajectory import Trajectory


class MonteCarloOffPolicyControl(BaseLearningAlgorithm):
    def __init__(self, env: Env, n_iter: int = 1000) -> None:
        self.env = env
        self.actions = list(range(env.action_space.n))

        self.q_values = defaultdict(lambda: np.random.rand(env.action_space.n))

        self.C = defaultdict(lambda: np.zeros(env.action_space.n))
        self.target_policy = lambda state: np.argmax(self.q_values[state])
        self.behavior_policy = lambda state: random.choice(self.actions)

        self.n_iter = n_iter
        self.discount_factor = 0.9
        self.policy_name = "MCPolicyControl(off-policy)"

    def get_policy(self):
        return self.target_policy

    def run_policy_iteration(self):
        trajectory = Trajectory()

        for _ in tqdm(range(self.n_iter)):
            trajectory.clear()
            obs, info = self.env.reset()
            done = False

            while not done:
                action = self.behavior_policy(state=obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                trajectory.record_step(state=obs, action=action, reward=reward)
                done = terminated or truncated
                obs = next_obs

            discounted_return = 0.0
            W = 1

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                discounted_return = self.discount_factor * discounted_return + reward
                self.C[state][action] = self.C[state][action] + W
                self.q_values[state][action] = self.q_values[state][action] + (
                    W / self.C[state][action]
                ) * (discounted_return - self.q_values[state][action])

                greedy_action = self.target_policy(state)

                if greedy_action != action:
                    break
                target_policy_action_prob = 1.0
                behavior_policy_action_prob = 1.0 / self.env.action_space.n
                W = W * (target_policy_action_prob / behavior_policy_action_prob)
