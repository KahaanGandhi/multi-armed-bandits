import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------------------#
# Local imports and data 
#------------------------#

from data_loader import load_subset
from agents import *
from environments import *

data_path = "/Users/kahaan/Desktop/multi-armed-bandits/data/"
subset = load_subset(data_path)

#-----------------#
# Directory setup
#-----------------#

# Establish base directory relative to current script location
# script_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# base_directory = os.path.normpath(os.path.join(script_location, '..', 'netflix-recommendations'))
# if not os.path.isdir(base_directory):
#    raise Exception(f"The directory {base_directory} does not exist.")
# if base_directory not in sys.path:
#    sys.path.append(base_directory)

#-------------------------------#
# Preprocess genres and ratings
#-------------------------------#

# Split genres into lists and expand each genre into seprate row
subset['Genres'] = subset['Genres'].str.split('|') 
df_exploded = subset.explode('Genres') 
unique_genres = df_exploded.Genres.unique()
genres = df_exploded["Genres"].unique().tolist()

# Process genre-specific rating distributions before passing to environments
unnormalized_distributions = {}
for genre in genres:
    subset = list(df_exploded[df_exploded["Genres"] == genre]["Rating"])
    rating_counts = {}
    for rating in range(1,6):
        rating_counts[rating] = subset.count(rating)
    unnormalized_distributions[genre] = rating_counts
    
# Determine niche genres based on threshold
movies_per_genre = df_exploded.groupby('Genres')['MovieID'].nunique()
ratings_per_genre = df_exploded.groupby('Genres')['Rating'].count()
threshold_movies = movies_per_genre.quantile(0.25)
threshold_ratings = ratings_per_genre.quantile(0.25)
niche_genres = movies_per_genre[movies_per_genre <= threshold_movies].index.tolist()

#----------------#
# Run simulation 
#----------------#

# Define step number and (optional) random seed
np.random.seed(777)
N = 1000

user1 = MultipleGenreEnjoyerEnvironment(genres, unnormalized_distributions, 1)
user2 = GenreEnjoyerEnvironment(genres, unnormalized_distributions, 2)
user3 = MultipleNicheGenreLoyalistEnvironment(genres, niche_genres, unnormalized_distributions, 3)
user4 = NicheGenreLoyalistEnvironment(genres, niche_genres, unnormalized_distributions, 4)
user5 = AverageViewerEnvironment(genres, unnormalized_distributions, 5)

user = user1

agent1 = EpsilonFirstAgent(genres, N, epsilon=0.1, environment=user)
agent2 = EpsilonGreedyAgent(genres, N, epsilon=0.1, environment=user)
agent3 = LinUCBAgent(genres, N, environment=user)
agent4 = EpsilonDecreasingAgent(genres, N, environment=user)

rewards1 = agent1.run()
rewards2 = agent2.run()
rewards3 = agent3.run()
rewards4 = agent4.run()

#-------------------------------#
# Evaluating agent performances
#-------------------------------#

# user.plot_distributions()
# agent1.analyze(plot_option="both")
# agent2.analyze(plot_option="both")
# agent3.analyze(plot_option="both")

# Helper function to calculate rolling average and cumulative average
def analyze_rewards(rewards, N):
    rewards = np.array(rewards)
    rolling_avg = pd.Series(rewards).rolling(window=max(10, int(N * 0.20))).mean() # Set sliding window size
    cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    return rolling_avg, cumulative_avg

rolling_avg1, cumulative_avg1 = analyze_rewards(rewards1, N)
rolling_avg2, cumulative_avg2 = analyze_rewards(rewards2, N)
rolling_avg3, cumulative_avg3 = analyze_rewards(rewards3, N)
rolling_avg4, cumulative_avg4 = analyze_rewards(rewards4, N)

# Plot rolling window average...
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
axes[0].plot(rolling_avg1, label='ε-First', color='maroon')
axes[0].plot(rolling_avg2, label='ε-Greedy', color='steelblue')
axes[0].plot(rolling_avg3, label='LinUCB', color='darkviolet')
axes[0].plot(rolling_avg4, label='ε-Decreasing', color='forestgreen')
axes[0].set_title('Rolling Window Average Reward')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Average Rating')
axes[0].legend()

# and cumulative average reward
axes[1].plot(cumulative_avg1, label='ε-First', color='maroon')
axes[1].plot(cumulative_avg2, label='ε-Greedy', color='steelblue')
axes[1].plot(cumulative_avg3, label='LinUCB', color='darkviolet')
axes[1].plot(cumulative_avg4, label='ε-Decreasing', color='forestgreen')
axes[1].set_title('Cumulative Average Reward')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Average Rating')
axes[1].legend()

plt.tight_layout()
plt.show()