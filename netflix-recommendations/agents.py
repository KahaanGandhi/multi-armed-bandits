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


class EpsilonFirstAgent:
    def __init__(self, genre_list, steps, epsilon=0.1, environment=None):
        
        # TODO: ADD A WAY TO RESET
        
        self.genre_list = genre_list
        
        # K-armed bandit problem: genres as arms and user as environment
        self.k = len(genre_list)
        self.N = steps
        self.epsilon = epsilon
        self.environment = environment
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.recent_rewards = []
        
    def run(self):
        rewards = []
        # Two-phase strategy: randomly explore for the first εN trials, then exploit best arm for the rest
        for i in range(self.N):
            # Exploration phase: select a random arm
            if i < (self.epsilon * self.N):
                arm = np.random.randint(0, self.k)
                reward = self.environment.get_reward(arm)
                self.arm_history[arm].append(reward)
            # Decision point: find arm with highest expected value
            elif i == int(self.epsilon * self.N):
                means = np.array([np.mean(rewards) if rewards else 0 for rewards in self.arm_history.values()])
                best_arm = np.argmax(means)
                reward = self.environment.get_reward(best_arm)
            # Exploitation phase: repeatedly select arm with highest expected value
            else:
                reward = self.environment.get_reward(best_arm)
            rewards.append(reward)
        self.recent_rewards = rewards  
        return rewards
    
    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, rewards=None, plot_option="both"):
        # Prepare Data
        rewards = np.array(self.recent_rewards)
        genres = np.array(self.genre_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
        if plot_option in ["both", "posterior"]:
            # Plot learned posteriors (Using KDE for a smooth curve)
            plt.figure(figsize=(14, 7))
            for idx, genre in enumerate(genres):
                data = self.arm_history[idx]
                if len(set(data)) > 1: 
                    sns.kdeplot(data, label=f'{genre}', color=colors[idx])
                else:
                    # Handle the single-value case by plotting a line
                    if data:
                        plt.axvline(x=data[0], label=f'{genre} (single value)', color=colors[idx], linestyle='--')
            plt.title('ε-first: Learned Posterior Distribution by Genre')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        
        if plot_option in ["both", "trends"]:
            # 2. Plot cumulative average reeward and sliding window average
            window_size = 200  
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            cumulative_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(14, 7))
            plt.plot(cumulative_average, label='Cumulative Average Reward', color='maroon', linestyle='--')
            plt.plot(rolling_avg, label=f'Rolling Average Reward (window size={window_size})', color='darkblue')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('ε-first: Reward Trends Over Time')
            plt.legend()
            plt.show()
            

class EpsilonGreedyAgent:
    def __init__(self, genre_list, steps, epsilon=0.1, environment=None):
        self.genre_list = genre_list
        
        # K-armed bandit problem: genres as arms and user as environment
        self.k = len(genre_list)
        self.N = steps
        self.epsilon = epsilon
        self.environment = environment
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.arm_EV = [0] * self.k
        self.arm_counts = [0] * self.k
        self.recent_rewards = []
        
    def run(self):
        rewards = []
        # Two-phase strategy: randomly explore for the first εN trials, then exploit best arm for the rest
        for i in range(self.N):
            p = np.random.random()
            if p < self.epsilon:
                # If exploration chosen, select a random arm 
                arm = np.random.randint(0, self.k)
            else:
                # If exploitation chosen, select arm with highest expected value
                arm = np.argmax(self.arm_EV)
            
            # Get the reward...
            reward = self.environment.get_reward(arm)
            rewards.append(reward)
            
            # and update the respective EVs
            n = self.arm_counts[arm] + 1
            self.arm_counts[arm] = n
            current_EV = self.arm_EV[arm]
            new_EV = current_EV + (reward - current_EV) / n
            self.arm_EV[arm] = new_EV
            self.arm_history[arm].append(reward)
            
        self.recent_rewards = rewards  
        return rewards
    
    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, rewards=None, plot_option="both"):
        # Prepare Data
        rewards = np.array(self.recent_rewards)
        genres = np.array(self.genre_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
        if plot_option in ["both", "posterior"]:
            # 1. Plot learned posteriors (Using KDE for a smooth curve)
            plt.figure(figsize=(14, 7))
            for idx, genre in enumerate(genres):
                data = self.arm_history[idx]
                if len(set(data)) > 1: 
                    sns.kdeplot(data, label=f'{genre}', color=colors[idx])
                else:
                    # Handle the single-value case by plotting a line
                    if data:
                        plt.axvline(x=data[0], label=f'{genre} (single value)', color=colors[idx], linestyle='--')
            plt.title('ε-greedy: Learned Posterior Distribution by Genre')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        
        if plot_option in ["both", "trends"]:
            # 2. Plot cumulative average reeward and sliding window average
            window_size = 200  
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            cumulative_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(14, 7))
            plt.plot(cumulative_average, label='Cumulative Average Reward', color='maroon', linestyle='--')
            plt.plot(rolling_avg, label=f'Rolling Average Reward (window size={window_size})', color='darkblue')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('ε-greedy: Reward Trends Over Time')
            plt.legend()
            plt.show()