import pytest
import os
import sys
import numpy as np
import warnings
from scipy.stats import ttest_ind

#-----------------------------------#
# Directory setup for local imports
#-----------------------------------#

current_dir = os.path.dirname(__file__)                      # Get directory where script is located
parent_dir = os.path.dirname(current_dir)                    # Get parent directory of current directory
src_path = os.path.abspath(os.path.join(parent_dir, 'src'))  # Construct absolute path to src
if src_path not in sys.path:
    sys.path.append(src_path)                                # Add src directory to path

try:
    from src.utils import *
    from src.environments import *
except ModuleNotFoundError:
    from utils import *
    from environments import *

#--------------------------#
# Load and preprocess data
#--------------------------#

DATA_PATH = "/Users/kahaan/Desktop/multi-armed-bandits/data/"
subset = load_subset(DATA_PATH)
genres, unnormalized_distributions, niche_genres = preprocess(subset, verbose=False)

#-----------------------------------------------------------------------------------------------------------#
# Test Suite for Environments (simulated users) 
# Verify that each accurately represents various user behaviors and preferences through simulated scenarios. 
#-----------------------------------------------------------------------------------------------------------#

def test_genre_enjoyer():
    # Instantiate the environment
    env = GenreEnjoyer(genres, unnormalized_distributions, user_id=777)
    
    # Check type and attributes of the instantiated object
    assert isinstance(env, GenreEnjoyer)
    assert isinstance(env.genre_list, list)
    assert isinstance(env.user_id, (int, type(None)))
    assert hasattr(env, 'favorite_genre')
    assert hasattr(env, 'reward_distributions')
    assert isinstance(env.reward_distributions, dict)
    
    # Check that genres were instantiated correctly
    assert len(env.reward_distributions) == len(genres)
    assert env.favorite_genre in env.genre_list
    
    # Two sample t-test to verify that rewards follow intended distribution, simulating 5000 interactions
    simulate_and_test(env, genres, unnormalized_distributions, trials=5000, alpha=0.05)
    
def test_niche_genre_loyalist():
    # Instantiate the environment
    env = NicheGenreLoyalist(genres, unnormalized_distributions, niche_genres, user_id=777)
    
    # Check type and attributes of the instantiated object
    assert isinstance(env, NicheGenreLoyalist)
    assert isinstance(env.genre_list, list)
    assert isinstance(env.user_id, (int, type(None)))
    assert isinstance(env.favorite_genre, str)
    assert isinstance(env.reward_distributions, dict)
    
    # Two sample t-test to verify that rewards follow intended distribution, simulating 5000 interactions
    simulate_and_test(env, genres, unnormalized_distributions, trials=5000, alpha=0.05)

def test_multiple_genre_enjoyer():
    # Instantiate the environment
    env = MultipleGenreEnjoyer(genres, unnormalized_distributions, user_id=777)
    
    # Check type and attributes of the instantiated object
    assert isinstance(env, MultipleGenreEnjoyer)
    assert isinstance(env.genre_list, list)
    assert isinstance(env.user_id, (int, type(None)))
    assert isinstance(env.favorite_genres, np.ndarray)
    assert isinstance(env.reward_distributions, dict)
    
    # Two sample t-test to verify that rewards follow intended distribution, simulating 5000 interactions
    simulate_and_test(env, genres, unnormalized_distributions, trials=5000, alpha=0.05)

def test_multiple_niche_genre_loyalist():
    env = MultipleNicheGenreLoyalist(genres, unnormalized_distributions, niche_genres, user_id=777)
    
    # Check type and attributes of the instantiated object
    assert isinstance(env, MultipleNicheGenreLoyalist)
    assert isinstance(env.genre_list, list)
    assert isinstance(env.user_id, (int, type(None)))
    assert isinstance(env.num_boosted_genres, int)
    assert isinstance(env.boosted_genres, np.ndarray)
    assert isinstance(env.reward_distributions, dict)
    
    # Check that genres were instantiated correctly
    assert len(env.reward_distributions) == len(genres)
    assert env.num_boosted_genres >= 1
    assert env.num_boosted_genres <= len(niche_genres)
    assert all(genre in niche_genres for genre in env.boosted_genres)
    
    # Two sample t-test to verify that rewards follow intended distribution, simulating 5000 interactions
    simulate_and_test(env, genres, unnormalized_distributions, trials=5000, alpha=0.05)

def test_average_viewer():
    # Instantiate the environment
    env = AverageViewer(genres, unnormalized_distributions, user_id=777)
    
    # Check type and attributes of the instantiated object
    assert isinstance(env, AverageViewer)
    assert isinstance(env.genre_list, list)
    assert isinstance(env.user_id, (int, type(None)))
    assert isinstance(env.reward_distributions, dict)
    
    # Two sample t-test to verify that rewards follow intended distribution, simulating 5000 interactions
    simulate_and_test(env, genres, unnormalized_distributions, trials=5000, alpha=0.05)

def simulate_and_test(env, genres, unnormalized_distributions, trials=5000, alpha=0.05):    
    # Expected probabilities: average of flattened the expected distributions 
    reward_distributions = env.create_reward_distributions(unnormalized_distributions)
    expected_probabilities = [np.mean(list(dist.values())) for dist in reward_distributions.values()]

    # Observed probabilities: average of distribution learned from 5000 trials
    observed_rewards = {rating: 0 for rating in range(1, 6)}
    for _ in range(trials):
        genre_index = np.random.randint(len(genres))
        reward = env.get_reward(genre_index)
        observed_rewards[reward] += 1
    observed_average_rewards = {rating: total / trials for rating, total in observed_rewards.items()}
    observed_probabilities = list(observed_average_rewards.values())
    
    # Perform two-sample t-test between observations and exepctations
    with warnings.catch_warnings():
        # Filter out a specific RunTime Warning which only occurs when the data is already nearly identical
        warnings.filterwarnings("ignore", 
                                message="Precision loss occurred in moment calculation due to catastrophic cancellation", 
                                category=RuntimeWarning)
        _, p_value = ttest_ind(observed_probabilities, expected_probabilities, equal_var=False)
    assert p_value > alpha    

if __name__ == "__main__":
    pytest.main([__file__])