from collections import defaultdict
from mc_suite.core.base_learning_algorithm import BaseLearningAlgorithm
from mc_suite.core.trajectory import Trajectory
from mc_suite.policies.base_policy import BasePolicy
from gymnasium import Env
from tqdm import tqdm


class MonteCarloEveryVisitPredictionIncremental(BaseLearningAlgorithm):
    """
    MonteCarloEveryVisitPredictionIncremental (also known as constant-alpha MC every visit) is as a prediction only algorithm.

    """

    def __init__(
        self,
        env: Env,
        policy: BasePolicy,
        step_size: float = 0.1,
        discount_factor: float = 0.9,
    ) -> None:
        super().__init__(name="MCEveryVisitPrediction")

        self.env = env
        self.policy = policy
        self.V = defaultdict(float)
        self.step_size = step_size
        self.discount_factor = discount_factor

    def get_policy(self):
        return self.policy

    def train(self, num_episodes: int, prediction_only: bool):
        if prediction_only == False:
            raise Exception("This is a prediction/evaluation only implementation.")

        trajectory = Trajectory()
        returns = defaultdict(float)

        for i in tqdm(range(num_episodes)):
            trajectory.clear()
            obs, info = self.env.reset()
            done = False

            while not done:
                action = self.policy.get_action(state=obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                trajectory.record_step(state=obs, action=action, reward=reward)
                done = terminated or truncated
                obs = next_obs

            discounted_return = 0.0

            for timestep in reversed(range(trajectory.get_trajectory_length())):
                state, action, reward = trajectory.get_step(timestep)
                discounted_return = self.discount_factor * discounted_return + reward

                returns[state] = discounted_return
                self.V[state] = self.V[state] + self.step_size * (
                    returns[state] - self.V[state]
                )

    def get_state_values(self):
        return self.V

    def get_state_action_values(self):
        raise Exception(
            f"{self.name} computes only the state values. Use get_state_values() method to get state values."
        )
