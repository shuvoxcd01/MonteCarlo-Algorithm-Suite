from collections import defaultdict
from typing import Optional
from gymnasium import Env
import numpy as np
from tqdm import tqdm

from mc_suite.core.base_learning_algorithm import BaseLearningAlgorithm
from mc_suite.core.util.episode_collector import collect_episode
from mc_suite.core.util.trajectory import Trajectory
from mc_suite.policies.base_policy import BasePolicy
from mc_suite.policies.random_policy import RandomPolicy
from mc_suite.policies.greedy.stochastic_start_greedy_policy import (
    StochasticStartGreedyPolicy,
)


class MonteCarloOffPolicySnB(BaseLearningAlgorithm):
    def __init__(
        self,
        env: Env,
        target_policy: Optional[BasePolicy] = None,
        behavior_policy: Optional[BasePolicy] = None,
        discount_factor: float = 0.9,
    ) -> None:
        super().__init__(name="MCPolicyControl(off-policy-SnB)")
        self.env = env
        self.num_actions = self.env.action_space.n
        self.actions = list(range(self.num_actions))

        self.q_values = defaultdict(lambda: np.random.rand(self.num_actions))

        self.C = defaultdict(lambda: np.zeros(self.num_actions))
        self.target_policy = (
            target_policy
            if target_policy is not None
            else StochasticStartGreedyPolicy(num_actions=self.num_actions)
        )
        self.behavior_policy = (
            behavior_policy
            if behavior_policy is not None
            else RandomPolicy(num_actions=self.num_actions)
        )

        self.discount_factor = discount_factor

    def get_policy(self):
        return self.target_policy

    def train(self, num_episodes: int, prediction_only: bool = False):
        trajectory = Trajectory()

        for _ in tqdm(range(num_episodes)):
            collect_episode(
                env=self.env, policy=self.behavior_policy, trajectory=trajectory
            )

            discounted_return = 0.0
            W = 1

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                discounted_return = self.discount_factor * discounted_return + reward
                self.C[state][action] = self.C[state][action] + W
                self.q_values[state][action] = self.q_values[state][action] + (
                    W / self.C[state][action]
                ) * (discounted_return - self.q_values[state][action])

                if not prediction_only:
                    greedy_action = np.argmax(self.q_values[state])
                    self.target_policy.update(state=state, action=greedy_action)

                    if greedy_action != action:
                        break

                target_policy_action_prob = self.target_policy.get_action_probs(
                    state=state, action=action
                )
                behavior_policy_action_prob = self.behavior_policy.get_action_probs(
                    state=state, action=action
                )

                if not prediction_only:
                    assert target_policy_action_prob == 1.0

                W = W * (target_policy_action_prob / behavior_policy_action_prob)

                if prediction_only and W == 0:
                    break

        self.target_policy.q_values = self.q_values

    def get_state_values(self):
        raise Exception(
            f"{self.name} computes only state-action values. Use get_state_action_values() to get state-action values."
        )

    def get_state_action_values(self):
        return self.q_values
