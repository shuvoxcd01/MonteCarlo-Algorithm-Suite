from abc import ABC, abstractmethod
import os
import dill

from src.data import DATA_DIR


class BaseLearningAlgorithm(ABC):
    @abstractmethod
    def get_policy(self):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def run_policy_iteration(self):
        raise NotImplementedError("This method must be overridden")

    def save_policy(self):
        policy = self.get_policy()
        policy_name = self.policy_name if self.policy_name else ""

        saved_policy_path = os.path.join(DATA_DIR, policy_name + "_saved_policy.pkl")

        serialized_policy = dill.dumps(policy)

        with open(saved_policy_path, "wb") as file:
            file.write(serialized_policy)

    @staticmethod
    def load_policy(saved_policy_path: str):
        with open(saved_policy_path, "rb") as file:
            policy = dill.loads(file.read())

        return policy
