import pytest
import pandas as pd
import sys
import os

#-----------------------------------#
# Directory setup for local imports
#-----------------------------------#

current_dir = os.path.dirname(__file__)     # Get directory where script is located
parent_dir = os.path.dirname(current_dir)   # Get parent directory
src_path = os.path.join(parent_dir, 'src')  # Constructs path to src directory
if src_path not in sys.path:
    sys.path.append(src_path)               # Add src directory to sys.path

# Try to import modules, handle failure gracefully
try:
    from src.utils import load_subset, preprocess
except ModuleNotFoundError:
    from utils import load_subset, preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data/')

#-----------------------------------------#
# Verify that data is loaded in correctly
#-----------------------------------------#

def test_load_subset():
    # Load the data
    subset = load_subset(DATA_PATH, verbose=False)
    assert not subset.empty, "Loaded dataset is empty"

    # Check DataFrame structure and integrity
    expected_columns = ['MovieID', 'UserID', 'Rating', 'Genres', 'Year', 'Title']
    assert all(column in subset.columns for column in expected_columns), "Missing expected columns"
    assert subset.isnull().sum().sum() == 0, "There are unexpected missing values in the data"
    
    # Check unique IDs
    assert subset['MovieID'].nunique() > 1000, "Number of unique MovieIDs is too low"
    assert subset['UserID'].nunique() > 25000, "Number of unique UserIDs is too low"

    # Basic statistics checks
    ratings = subset['Rating']
    assert ratings.min() >= 1 and ratings.max() <= 5, "Rating values are outside expected range 1-5"
    assert len(ratings[ratings.isin([1, 2, 3, 4, 5])]) == len(ratings), "Ratings contain invalid values"
    assert subset['Rating'].mean() > 2 and subset['Rating'].mean() < 5, "Average rating is out of normal bounds"

#--------------------------------------------#
# Verify that data is preprocessed correctly
#--------------------------------------------#

def test_preprocess():
    # Assuming subset has been loaded correctly
    subset = load_subset(DATA_PATH)
    genres, unnormalized_distributions, niche_genres = preprocess(subset, verbose=False)
    
    # Validate genres and niche genres content
    assert len(genres) > 20, "Expected more unique genres"
    assert len(niche_genres) < len(genres), "Niche genres should be a subset of all genres"
    assert all(genre in genres for genre in niche_genres), "Niche genres list contains invalid genres"
    
    # Validation of rating distributions
    for genre, distribution in unnormalized_distributions.items():
        total_ratings_count = sum(distribution.values())
        mask = subset['Genres'].apply(lambda x: genre in x)
        expected_count = subset[mask]['Rating'].count()
        assert total_ratings_count == expected_count, f"Total ratings count mismatch for {genre}"

    # Additional consistency checks for rating distribution
    if 'Documentary' in unnormalized_distributions:
        assert sum(unnormalized_distributions['Documentary'].values()) > 5000, "Unexpected low number of ratings for 'Documentary'"

if __name__ == "__main__":
    pytest.main([__file__])
