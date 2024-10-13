from collections import defaultdict
from gymnasium import Env
import numpy as np
from tqdm import tqdm
import random

from src.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from src.algorithms.trajectory import Trajectory



class MonteCarloES(BaseLearningAlgorithm):
    def __init__(self, env: Env, n_iter: int = 1000) -> None:
        self.env = env
        self.actions = list(range(env.action_space.n))
        self.policy = defaultdict(lambda: random.choice(self.actions))
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.n_iter = n_iter
        self.discount_factor = 0.9

        self.policy_name = "MCES"

    
    def get_policy(self):
        return lambda state: self.policy[state]
    
    def run_policy_iteration(self):
        trajectory = Trajectory()
        returns = defaultdict(list)

        for _ in tqdm(range(self.n_iter)):
            trajectory.clear()
            obs, info = self.env.reset()
            action = random.choice(self.actions)
            done = False

            while not done:
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                trajectory.record_step(state=obs, action=action, reward=reward)
                done = terminated or truncated
                obs = next_obs
                action = self.policy[obs]

            discounted_return = 0.0

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                state_action = (state, action)
                discounted_return = self.discount_factor * discounted_return + reward

                if not trajectory.check_state_action_appearance_before_timestep(
                    state_action, timestep
                ):
                    returns[state_action].append(discounted_return)
                    q_value = sum(returns[state_action]) / len(returns[state_action])
                    self.q_values[state][action] = q_value
                    self.policy[state] = np.argmax(self.q_values[state])

