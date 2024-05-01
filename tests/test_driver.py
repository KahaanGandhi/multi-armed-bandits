import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

#-----------------------------------#
# Directory setup for local imports
#-----------------------------------#

current_dir = os.path.dirname(__file__)     # Get directory where script is located
parent_dir = os.path.dirname(current_dir)   # Get parent directory of current directory
src_path = os.path.join(parent_dir, 'src')  # Construct absolute path to src
if src_path not in sys.path:
    sys.path.append(src_path)               # Add src directory to path

# Try to import modules, handle failure gracefully
try: 
    from src.environments import *
    from src.agents import *
    from src.utils import *
except ModuleNotFoundError:
    from environments import *
    from agents import *
    from utils import *

#-----------------------------------#
# Simulation settings and parameters
#-----------------------------------#

random_seed = 1
total_users = 3
N = 10000

# Create an environment instance based on the class and provided parameters
def create(env_class, genres, distributions, niche_genres, index):
    if env_class in [MultipleNicheGenreLoyalist, NicheGenreLoyalist]:
        return env_class(genres, distributions, niche_genres, index)
    else:
        return env_class(genres, distributions, index)

# Run each agent in the given environment and collect cumulative average rewards
def run(environment, agents, steps=N):
    results = {}
    for agent_name, agent_class in agents.items():
        agent = agent_class(environment.genre_list, steps, environment=environment)
        rewards = agent.run()
        results[agent_name] = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    return results

#----------------------------------------------------------#
# Pytest fixtures for setting up the data and environments
#----------------------------------------------------------#

# Load data from specified path and preprocess it for simulation
@pytest.fixture(scope="module")
def setup_data():
    np.random.seed(random_seed)
    BASE_DIR = os.path.dirname(os.getcwd())
    DATA_PATH = os.path.join(BASE_DIR, 'data/')
    subset = load_subset(DATA_PATH)
    genres, unnormalized_distributions, niche_genres = preprocess(subset, verbose=False)
    return genres, unnormalized_distributions, niche_genres

# Create a list of environment objects based on the setup data
@pytest.fixture(scope="module")
def setup_environments(setup_data):
    genres, distributions, niche_genres = setup_data
    environments = [
        (MultipleNicheGenreLoyalist, int(0.22 * total_users)),
        (MultipleGenreEnjoyer, int(0.44 * total_users)),
        (GenreEnjoyer, int(0.06 * total_users)),
        (NicheGenreLoyalist, int(0.10 * total_users)),
        (AverageViewer, int(0.18 * total_users)),
    ]
    env_objects = []
    for env_class, count in environments:
        for i in range(count):
            env = create(env_class, genres, distributions, niche_genres, i+1)
            env_objects.append(env)
    return env_objects

# Run all agents across all environments and compile results
@pytest.fixture(scope="module")
def run_agents(setup_environments):
    environments = setup_environments
    agents = {
        'Dirichlet Forest Sampling': DirichletForestSampling,
        'Deep Q-Network': DeepQNetwork,
        'Advantage Actor-Critic': AdvantageActorCritic,
        'ε-first': EpsilonFirst,
        'ε-greedy': EpsilonGreedy,
        'ε-decreasing': EpsilonDecreasing,
        'LinUCB': LinUCB,
        'A/B Testing': ABTesting,
    }
    overall_results = {name: [] for name in agents}
    with patch("matplotlib.pyplot.show"):
        for environment in environments:
            results = run(environment, agents)
            for name in agents:
                overall_results[name].append(results[name])
    return overall_results

# Ensure that agents perform as expected against historical results and converge
def test_agent_performance(run_agents):
    results = run_agents
    
    # Compare best and worst performing agents against known results for fixed random seed
    final_averages = {agent: np.mean([run[-1] for run in results[agent]]) for agent in results}
    assert final_averages['Dirichlet Forest Sampling'] > final_averages['Deep Q-Network']
    assert final_averages['Dirichlet Forest Sampling'] > final_averages['A/B Testing']
    assert final_averages['LinUCB'] > final_averages['Deep Q-Network']
    assert final_averages['LinUCB'] > final_averages['A/B Testing']
    
    # Convergence criterion
    final_changes = {agent: np.mean([abs(run[-1] - run[-2]) for run in results[agent]]) for agent in results}
    for change in final_changes.values():
        assert change < 0.01

# if __name__ == "__main__":
#     pytest.main([__file__])
