import gymnasium as gym
from mc_suite.core.monte_carlo_exploring_start import MonteCarloES
from mc_suite.core.monte_carlo_off_policy import MonteCarloOffPolicy
from mc_suite.core.monte_carlo_off_policy_snb import MonteCarloOffPolicySnB

import click

algorithm_map = {
    "mc_es": MonteCarloES,
    "mc_off_policy": MonteCarloOffPolicy,
    "mc_off_policy_snb": MonteCarloOffPolicySnB,
}

n_episodes = 10000
env = gym.make("Blackjack-v1", sab=True)
agent = MonteCarloOffPolicy(env=env)
agent.train(num_episodes=n_episodes)


@click.command()
@click.option("--env", required=True, help="Name of the gymnasium environment")
@click.option(
    "--algorithm",
    required=True,
    help="Name of algorithm to use. Available options are: mc_es, mc_off_policy, mc_off_policy_snb",
)
@click.option("--num_episodes", default=10000, help="Number to episodes to train for")
def run_poicy_optimization(env, algorithm, num_episodes):
    env = gym.make(env)

    algorithm = algorithm.get(algorithm, None)

    if algorithm is None:
        raise Exception("Invalid algorithm name")

    agent = algorithm(env=env)
    agent.optimize_policy(num_episodes=num_episodes)

    optimized_policy = agent.get_policy()

    return optimized_policy

if __name__ == "__main__":
    run_poicy_optimization()