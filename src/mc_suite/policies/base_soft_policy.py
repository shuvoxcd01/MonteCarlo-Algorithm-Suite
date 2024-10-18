from abc import abstractmethod

from mc_suite.policies.base_policy import BasePolicy


class BaseSoftPolicy(BasePolicy):
    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError("This method must be overridden")
    
    @abstractmethod
    def get_action_deterministic(self, state):
        raise NotImplementedError("This method must be overridden")
    
    @abstractmethod
    def get_action_probs(self, state, action):
        raise NotImplementedError("This method must be overridden")
    
    @abstractmethod
    def update(self, state, action):
        raise NotImplementedError("This method must be overridden")
