import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

script_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
base_directory = os.path.normpath(os.path.join(script_location, '..', 'netflix-recommendations'))
if not os.path.isdir(base_directory):
    raise Exception(f"The directory {base_directory} does not exist.")
if base_directory not in sys.path:
    sys.path.append(base_directory)

from data_loader import load_subset
from agents import EpsilonGreedyAgent, EpsilonFirstAgent, LinUCBAgent
from environments import GenreEnjoyerEnvironment

data_path = "/Users/kahaan/Desktop/multi-armed-bandits/netflix-recommendations/data/"
subset = load_subset(data_path)

# Split genres into lists and expand each genre into seprate row
subset['Genres'] = subset['Genres'].str.split('|') 
df_exploded = subset.explode('Genres') 
unique_genres = df_exploded.Genres.unique()
genres = df_exploded["Genres"].unique().tolist()

# Process genre-specific rating distributions before passing to enviornments
unnormalized_distributions = {}
for genre in genres:
    subset = list(df_exploded[df_exploded["Genres"] == genre]["Rating"])
    rating_counts = {}
    for rating in range(1,6):
        rating_counts[rating] = subset.count(rating)
    unnormalized_distributions[genre] = rating_counts
    
np.random.seed(777)

viewer1 = GenreEnjoyerEnvironment(genres, unnormalized_distributions, 1)
viewer1.plot_distributions()

agent1 = EpsilonFirstAgent(genres, 10000, epsilon=0.1, environment=viewer1)
agent1.run()
agent1.analyze(plot_option="both")

np.random.seed(7)

viewer2 = GenreEnjoyerEnvironment(genres, unnormalized_distributions, 2)
viewer2.plot_distributions()

agent2 = EpsilonGreedyAgent(genres, 10000, epsilon=0.1, environment=viewer2)
agent2.run()
agent2.analyze(plot_option="both")

viewer3 = GenreEnjoyerEnvironment(genres, unnormalized_distributions, 3)
viewer3.plot_distributions()

agent3 = LinUCBAgent(genres, 10000, environment=viewer3)
agent3.run()
agent3.analyze(plot_option="both")
