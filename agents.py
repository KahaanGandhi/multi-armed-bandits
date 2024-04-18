import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#---------------------------------------#
# LinUCB (upper confidence bound) agent
#---------------------------------------#

class LinUCBAgent:
    def __init__(self, genre_list, steps, environment=None):
        self.genre_list = genre_list
        
        # K-armed bandit problem: genres as arms and user as environment
        self.k = len(genre_list)
        self.N = steps
        self.environment = environment
        self.arm_EV = [0] * self.k
        self.arm_counts = [0] * self.k
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}

    def run(self):
        rewards = []
        # Selects arm with highest predicted reward + confidence, adapting to context to optimize decisions over time
        for i in range(self.N):
            ucb_values = []
            # Calculate expected value and confidence interval for each arm to define upper confidence bound
            for arm in range(self.k):
                if self.arm_counts[arm] > 0:
                    EV = self.arm_EV[arm]
                    confidence = np.sqrt(2 * np.log(i)) / self.arm_counts[arm]
                    ucb_value = EV + confidence
                else:
                    # Ensure that each arm is explored at least once
                    ucb_value = np.inf
                ucb_values.append(ucb_value)
            # Select arm with highest UCB value...    
            chosen_arm = np.argmax(ucb_values)
            reward = self.environment.get_reward(chosen_arm)
            self.arm_history[chosen_arm].append(reward)
            rewards.append(reward)
            self.arm_counts[chosen_arm] += 1
            # and update the corresponding EV
            old_EV = self.arm_EV[chosen_arm]
            n = self.arm_counts[chosen_arm]
            self.arm_EV[chosen_arm] = (old_EV * (n - 1) + reward) / n
        self.recent_rewards = rewards 
        return rewards    

    
    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, rewards=None, plot_option="both"):
        # Prepare data
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
            plt.title('LinUCB: Learned Posterior Distribution by Genre')
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
            plt.title('LinUCB: Reward Trends Over Time')
            plt.legend()
            plt.show()

#---------------#
# ε-first agent
#---------------#

class EpsilonFirstAgent:
    def __init__(self, genre_list, steps, epsilon=0.1, environment=None):
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
        # Prepare data
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
            
#----------------#
# ε-greedy agent
#----------------#

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
        # Prepare data
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
            window_size = max(10, int(self.N * 0.02))  
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