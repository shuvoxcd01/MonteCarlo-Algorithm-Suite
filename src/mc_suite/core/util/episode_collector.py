from mc_suite.core.util.trajectory import Trajectory
from mc_suite.policies.base_policy import BasePolicy
from gymnasium import Env


def collect_episode(env:Env, policy:BasePolicy, trajectory:Trajectory):
    trajectory.clear()
    obs, info = env.reset()
    done = False

    while not done:
        action = policy.get_action(state=obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        trajectory.record_step(state=obs, action=action, reward=reward)
        done = terminated or truncated
        obs = next_obs