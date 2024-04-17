import sys
sys.path.append("../")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from data_loader import load_subset

data_path = "/Users/kahaan/Desktop/multi-armed-bandits/netflix-recommendations/data/"

subset = load_subset(data_path)

# Split genres into lists and expand each genre into seprate row
subset['Genres'] = subset['Genres'].str.split('|') 
df_exploded = subset.explode('Genres') 
unique_genres = df_exploded.Genres.unique()

genres = df_exploded["Genres"].unique().tolist()

unnormalized_distributions = {}
for genre in genres:
    subset = list(df_exploded[df_exploded["Genres"] == genre]["Rating"])
    rating_counts = {}
    for rating in range(1,6):
        rating_counts[rating] = subset.count(rating)
    unnormalized_distributions[genre] = rating_counts
    

class GenreEnjoyerEnvironment:
    def __init__(self, genre_list, unnormalized_distributions, user_id):
        self.genre_list = genre_list
        self.user_id = user_id
        self.favorite_genre = np.random.choice(genre_list)  # Randomly select a favorite genre
        self.reward_distributions = self.create_reward_distributions(unnormalized_distributions)

    # Create stochastic and stationary reward distributions for each genre, biased towards favorite
    def create_reward_distributions(self, unnormalized_distributions):
        normalized_distributions = {}
        for genre in self.genre_list:
            rating_counts = unnormalized_distributions[genre].copy()
            if genre == self.favorite_genre:
                # Apply a bias to ratings 4 and 5 for the favorite genre
                bias_factor = 5
                rating_counts[4] *= bias_factor
                rating_counts[5] *= bias_factor

            # Normalize counts to create a probability distribution
            total_counts = sum(rating_counts.values())
            normalized_counts = {}
            for rating, count in rating_counts.items():
                normalized_counts[rating] = count / total_counts
            normalized_distributions[genre] = normalized_counts
        
        return normalized_distributions
    
    # Sample for reward distribution for selected genre
    def get_reward(self, genre_index):
        genre = self.genre_list[genre_index]
        ratings, probabilities = zip(*self.reward_distributions[genre].items()) # TODO: review this syntax
        reward = np.random.choice(ratings, p=probabilities)
        return reward
        
    # Plot the normalized rating distribution for each genre
    def plot_distributions(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.genre_list)))
        
        for genre, color in zip(self.genre_list, colors):
            if genre in self.reward_distributions:
                ratings = list(self.reward_distributions[genre].keys())
                probabilities = list(self.reward_distributions[genre].values())
                if genre == self.favorite_genre:
                    plt.plot(ratings, probabilities, '-o', label=genre + " (Fav)", color="red")
                else:
                    plt.plot(ratings, probabilities, '-o', label=genre, color=color)
        
        plt.title('Normalized Rating Distribution for Genre Enjoyer')
        plt.xlabel('Rating')
        plt.ylabel('Proportion')
        plt.xticks(range(1, 6))
        plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()