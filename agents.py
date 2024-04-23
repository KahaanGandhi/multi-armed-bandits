import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#--------------------------------------------------------------------------------------------------------------------#
# Dirichlet Sampling (Multinomial Thompson Sampling)
# Uses Thompson Sampling with multinomial reward distribution modeling. Based on papers below.
# Riou and Honda, 2020, "Bandit Algorithms Based on Thompson Sampling for Bounded Reward Distributions"
# Baudry, Saux, Maillard, 2021, "From Optimality to Robustness: Dirichlet Sampling Strategies in Stochastic Bandits"
#--------------------------------------------------------------------------------------------------------------------#

class DirichletSamplingAgent:
    def __init__(self, genre_list, steps, environment=None):
        self.k = len(genre_list)
        
        # K-armed bandit problem: genres as arms and user as environment
        self.N = steps  
        self.environment = environment
        self.dirichlet_params = np.ones((self.k, 5))  # Initialize Dirichlet parameters with uniform priors
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}  

    # At each step, sample from Dirichlet distribution to maximize EV based on iteratively updated reward probabilities
    def run(self):
        rewards = []
        for i in range(self.N):
            sampled_probs = [np.random.dirichlet(params) for params in self.dirichlet_params]  # Sample a probability vector...
            expected_rewards = [np.dot(probs, np.arange(1, 6)) for probs in sampled_probs]     # use it to calculate EV...
            chosen_arm = np.argmax(expected_rewards)                                           # and select the highest EV arm
            
            # Retrieve reward and update history 
            reward = self.environment.get_reward(chosen_arm)
            rewards.append(reward)
            self.arm_history[chosen_arm].append(reward)
            
            # Weight higher ratings more, and boost well-performing genres every 100 steps
            self.dirichlet_params[chosen_arm][reward - 1] += (1 + reward / 5)
            if (i + 1) % 100 == 0:
                self.boost()
    
        self.recent_rewards = rewards
        return rewards
    
    # Boost parameters for genres with average rewards greater than 4
    def boost(self):
        average_rewards = [np.mean(history) if history else 0 for history in self.arm_history.values()]
        for i, avg in enumerate(average_rewards):
            if avg > 4: 
                self.dirichlet_params[i] += np.array([0, 0, 1, 1, 2])
    
    # Clear history and reset Reset Dirichlet parameters to uniform priors to start a new run
    def reset(self):
        self.dirichlet_params = np.ones((self.k, 5))  
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.recent_rewards = []

    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, plot_option="both"):
        # Prepare data
        rewards = np.array(self.recent_rewards)
        genres = np.array(self.genre_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
        if plot_option in ["both", "posterior"]:
            plt.figure(figsize=(14, 7))
            for idx, genre in enumerate(genres):
                data = np.array(self.arm_history[idx])
                # Handle the single-value case by plotting a line
                if len(np.unique(data)) > 1:
                    sns.kdeplot(data, label=f'{genre}', color=colors[idx])
                else:
                    plt.axvline(x=data[0] if len(data) > 0 else 0, label=f'{genre} (single value)', linestyle='--', color=colors[idx])
            plt.title('Multinomial Thompson Sampling: Learned Posterior Distribution by Genre')
            plt.legend(loc='upper right')
            plt.show()
        # 2. Plot cumulative average reeward and sliding window average
        if plot_option in ["both", "trends"]:
            window_size = max(10, int(self.N * 0.02))
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            cumulative_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(14, 7))
            plt.plot(cumulative_average, label='Cumulative Average Reward', linestyle='--', color='maroon')
            plt.plot(rolling_avg, label='Rolling Average Reward (window size={window_size})', color='darkblue')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('Multinomial Thompson Sampling: Reward Trends Over Time')
            plt.legend()
            plt.show()

#----------------------------------------------------------------#
# LinUCB (upper confidence bound)
# Uses linear models with upper confidence bounds for decisions.
#----------------------------------------------------------------#

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

    # Selects arm with highest predicted reward + confidence, adapting to context to optimize decisions over time
    def run(self):
        rewards = []
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

    # Clear history and statistics to start a new run
    def reset(self):
        self.arm_EV = [0] * self.k
        self.arm_counts = [0] * self.k
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        
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

#-------------------------------------------------------------------------------#
# ε-decreasing 
# Balances exploitation and exploration; adaptively decreases exploration rate.
#-------------------------------------------------------------------------------#

class EpsilonDecreasingAgent:
    def __init__(self, genre_list, steps, environment=None):
        self.genre_list = genre_list
        self.k = len(genre_list)
        self.N = steps
        self.initial_epsilon = 0.99  # Starting epsilon near 1 for maximum initial exploration
        self.minimum_epsilon = 0.01  # Minimum epsilon to maintain some exploration
        self.environment = environment
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.arm_EV = [0] * self.k
        self.arm_counts = [0] * self.k
        self.recent_rewards = []
    
    # ε decreases over time: starts by exploring frequently, gradually shifts to exploiting best arm    
    def run(self):
        rewards = []
        for i in range(self.N):
            # Exponential decay of ε
            current_epsilon = max(self.initial_epsilon * np.exp(-i / (self.N / 5)), self.minimum_epsilon)
            p = np.random.random()
            if p < current_epsilon:
                # Explore: choose a random arm
                arm = np.random.randint(0, self.k)
            else:
                # Exploit: choose the best known arm
                arm = np.argmax(self.arm_EV) 
            reward = self.environment.get_reward(arm)
            rewards.append(reward)
            # Update the arm's expected value (EV) and history
            n = self.arm_counts[arm] + 1
            self.arm_counts[arm] = n
            current_EV = self.arm_EV[arm]
            new_EV = current_EV + (reward - current_EV) / n
            self.arm_EV[arm] = new_EV
            self.arm_history[arm].append(reward)
        self.recent_rewards = rewards  
        return rewards

    # Clear history and statistics to start a new run
    def reset(self):
        self.arm_EV = [0] * self.k
        self.arm_counts = [0] * self.k
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}

    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, rewards=None, plot_option="both"):
        # Prepare data
        rewards = np.array(self.recent_rewards) if rewards is None else np.array(rewards)
        genres = np.array(self.genre_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
        if plot_option in ["both", "posterior"]:
            plt.figure(figsize=(14, 7))
            for idx, genre in enumerate(genres):
                data = self.arm_history[idx]
                if len(set(data)) > 1: 
                    sns.kdeplot(data, label=f'{genre}', color=colors[idx])
                else:
                    if data:
                        plt.axvline(x=data[0], label=f'{genre} (single value)', color=colors[idx], linestyle='--')
            plt.title('ε-decreasing: Learned Posterior Distribution by Genre')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        # 2. Plot cumulative average reeward and sliding window average  
        if plot_option in ["both", "trends"]:
            window_size = max(10, int(self.N * 0.02))  
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            cumulative_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(14, 7))
            plt.plot(cumulative_average, label='Cumulative Average Reward', color='maroon', linestyle='--')
            plt.plot(rolling_avg, label=f'Rolling Average Reward (window size={window_size})', color='darkblue')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('ε-decreasing: Reward Trends Over Time')
            plt.legend()
            plt.show()

#---------------------------------------------------------------------#
# ε-first
# Explores uniformly early on, then exclusively exploits best option.
#---------------------------------------------------------------------#

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

    # Two-phase strategy: randomly explore for the first εN steps, then exploit best arm for the rest
    def run(self):
        rewards = []
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
    
    # Clear history and statistics to start a new run
    def reset(self):
        self.arm_EV = [0] * self.k
        self.arm_counts = [0] * self.k
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
    
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
            
#------------------------------------------------------------------#
# ε-greedy
# Randomly explores but primarily exploits the best-known option.
#------------------------------------------------------------------#

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
     
    # At each step, 1-ε probability to explore random arm, ε probability to exploit best arm 
    def run(self):
        rewards = []
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
    
    # Clear history and statistics to start a new run
    def reset(self):
        self.arm_EV = [0] * self.k
        self.arm_counts = [0] * self.k
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
    
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

#------------------------------------------------------------------#
# A/B testing
# Compares multiple strategies to identify the most effective one.
#------------------------------------------------------------------#

class ABTestingAgent:
    def __init__(self, genre_list, steps, environment):
        self.genre_list = genre_list
        
        # K-armed bandit problem: genres as arms and user as environment
        self.k = len(genre_list)
        self.N = steps
        self.environment = environment
        self.arm_rewards = np.zeros(self.k)
        self.arm_counts = np.zeros(self.k)
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        
    # A/B testing strategy, designed for a cold-start scenario
    # Initially explores all arms equally, then focuses on more promising ones 
    def run(self):
        rewards = [] 
        exploration_steps = int(self.N * 0.2) 
        # Phase 1: initial uniform exploration for 20% of total steps
        for t in range(exploration_steps):
            chosen_arm = t % self.k
            reward = self.environment.get_reward(chosen_arm)
            rewards.append(reward)
            self.arm_history[chosen_arm].append(reward)
            self.arm_rewards[chosen_arm] += reward
            self.arm_counts[chosen_arm] += 1
        # Phase 2: Exploit the best arms for 80%, continue to explore for 20%
        for t in range(exploration_steps, self.N):
            if np.random.rand() < 0.8:
                best_arm = np.argmax(self.arm_rewards / (self.arm_counts + 1e-10))
                chosen_arm = best_arm
            else:
                chosen_arm = np.random.randint(self.k)
            reward = self.environment.get_reward(chosen_arm)
            rewards.append(reward)
            self.arm_history[chosen_arm].append(reward)
            self.arm_rewards[chosen_arm] += reward
            self.arm_counts[chosen_arm] += 1
        self.recent_rewards = rewards
        return rewards

    # Clear history and statistics to start a new run
    def reset(self):
        self.arm_rewards = np.zeros(self.k)
        self.arm_counts = np.zeros(self.k)
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        
    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, plot_option="both"):
        # Prepare data
        rewards = np.array(self.recent_rewards)
        genres = np.array(self.genre_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
        if plot_option in ["both", "posterior"]:
            plt.figure(figsize=(14, 7))
            for idx, genre in enumerate(genres):
                data = np.array(self.arm_history[idx])
                # Handle the single-value case by plotting a line
                if len(np.unique(data)) > 1:
                    sns.kdeplot(data, label=f'{genre}', color=colors[idx])
                else:
                    plt.axvline(x=data[0] if len(data) > 0 else 0, label=f'{genre} (single value)', linestyle='--', color=colors[idx])
            plt.title('A/B Testing: Learned Posterior Distribution by Genre')
            plt.legend(loc='upper right')
            plt.show()
        # 2. Plot cumulative average reeward and sliding window average
        if plot_option in ["both", "trends"]:
            window_size = max(10, int(self.N * 0.02))
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            cumulative_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(14, 7))
            plt.plot(cumulative_average, label='Cumulative Average Reward', linestyle='--', color='maroon')
            plt.plot(rolling_avg, label='Rolling Average Reward (window size={window_size})', color='darkblue')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('A/B Testing: Reward Trends Over Time')
            plt.legend()
            plt.show()
            
#--------------------------------------------------------------------------#
# ε-decreasing hybrid
# Mixes Thompson Sampling and UCB strategies based on decaying probability.
#--------------------------------------------------------------------------#

class EpsilonDecreasingHybridAgent:
    def __init__(self, genre_list, steps, epsilon=0.2, environment=None):
        self.k = len(genre_list)
        
        # K-armed bandit problem: genres as arms and user as environment
        self.N = steps
        self.epsilon = epsilon
        self.environment = environment
        self.dirichlet_params = np.ones((self.k, 5))
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.arm_counts = np.zeros(self.k)

    # At each step, first decay ε, then ε probability of LinUCB and 1-ε probability of Multinomial Thompson sampling 
    def run(self):
        rewards = []
        for i in range(self.N):
            # Dynamic exploration factor that decreases as the number of steps increases
            exploration_prob = self.epsilon * np.log(1 + i) / np.log(1 + self.N)
            if np.random.rand() < exploration_prob:
                # Use UCB based on the dynamic exploration factor...
                confidence_bounds = [np.mean(self.dirichlet_params[arm]) + np.sqrt(2 * np.log(i+1) / (self.arm_counts[arm] + 1e-10)) for arm in range(self.k)]
                chosen_arm = np.argmax(confidence_bounds)
            else:
                # otherwise use Multinomial Thompson Sampling
                sampled_probs = [np.random.dirichlet(params) for params in self.dirichlet_params] 
                expected_rewards = [np.dot(probs, np.arange(1, 6)) for probs in sampled_probs]
                chosen_arm = np.argmax(expected_rewards)
            reward = self.environment.get_reward(chosen_arm)
            rewards.append(reward)
            self.recent_rewards.append(reward)
            self.arm_history[chosen_arm].append(reward)
            self.dirichlet_params[chosen_arm][reward - 1] += 1
            self.arm_counts[chosen_arm] += 1
        self.recent_rewards = rewards
        return rewards
    
    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, plot_option="both"):
        # Prepare data
        rewards = np.array(self.recent_rewards)
        genres = np.array(self.genre_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
        if plot_option in ["both", "posterior"]:
            plt.figure(figsize=(14, 7))
            for idx, genre in enumerate(genres):
                data = np.array(self.arm_history[idx])
                # Handle the single-value case by plotting a line
                if len(np.unique(data)) > 1:
                    sns.kdeplot(data, label=f'{genre}', color=colors[idx])
                else:
                    plt.axvline(x=data[0] if len(data) > 0 else 0, label=f'{genre} (single value)', linestyle='--', color=colors[idx])
            plt.title('ε-decreasing Hybrid: Learned Posterior Distribution by Genre')
            plt.legend(loc='upper right')
            plt.show()
        # 2. Plot cumulative average reeward and sliding window average
        if plot_option in ["both", "trends"]:
            window_size = max(10, int(self.N * 0.02))
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            cumulative_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(14, 7))
            plt.plot(cumulative_average, label='Cumulative Average Reward', linestyle='--', color='maroon')
            plt.plot(rolling_avg, label='Rolling Average Reward (window size={window_size})', color='darkblue')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('ε-decreasing Hybrid: Reward Trends Over Time')
            plt.legend()
            plt.show()

#-----------------------------------------------------------------------#
# ε-greedy hybrid
# Mixes Thompson Sampling and UCB strategies based on fixed probability.
#-----------------------------------------------------------------------#
class EpsilonGreedyHybridAgent:
    def __init__(self, genre_list, steps, epsilon=0.1, environment=None):
        self.k = len(genre_list)
        
        # K-armed bandit problem: genres as arms and user as environment
        self.N = steps
        self.epsilon = epsilon
        self.environment = environment
        self.dirichlet_params = np.ones((self.k, 5))
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.arm_counts = np.zeros(self.k)

    # At each step, ε probability to select LinUCB, 1-ε probability to select Multinomial Thompson sampling
    def run(self):
        rewards = []
        for i in range(self.N):
            if np.random.rand() < self.epsilon:
                # LinUCB: select arm with highest predicted reward + confidence,
                confidence_bounds = [np.mean(self.dirichlet_params[arm]) + np.sqrt(2 * np.log(i+1) / (self.arm_counts[arm] + 1e-10)) for arm in range(self.k)]
                chosen_arm = np.argmax(confidence_bounds)
            else:
                # Multinomial Thompson Sampling: sample a probability vector to calculate EV, then select the highest EV arm
                sampled_probs = [np.random.dirichlet(params) for params in self.dirichlet_params] 
                expected_rewards = [np.dot(probs, np.arange(1, 6)) for probs in sampled_probs]
                chosen_arm = np.argmax(expected_rewards)
            
            reward = self.environment.get_reward(chosen_arm)
            rewards.append(reward)
            self.recent_rewards.append(reward)
            self.arm_history[chosen_arm].append(reward)
            self.dirichlet_params[chosen_arm][reward - 1] += 1
            self.arm_counts[chosen_arm] += 1
        self.recent_rewards = rewards
        return rewards

    def reset(self):
        self.dirichlet_params = np.ones((self.k, 5))
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.arm_counts = np.zeros(self.k)
        
    # Generate plots and diagnostics to evaluate agent performance 
    def analyze(self, plot_option="both"):
        # Prepare data
        rewards = np.array(self.recent_rewards)
        genres = np.array(self.genre_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
        if plot_option in ["both", "posterior"]:
            plt.figure(figsize=(14, 7))
            for idx, genre in enumerate(genres):
                data = np.array(self.arm_history[idx])
                # Handle the single-value case by plotting a line
                if len(np.unique(data)) > 1:
                    sns.kdeplot(data, label=f'{genre}', color=colors[idx])
                else:
                    plt.axvline(x=data[0] if len(data) > 0 else 0, label=f'{genre} (single value)', linestyle='--', color=colors[idx])
            plt.title('ε-greedy Hybrid: Learned Posterior Distribution by Genre')
            plt.legend(loc='upper right')
            plt.show()
        # 2. Plot cumulative average reeward and sliding window average
        if plot_option in ["both", "trends"]:
            window_size = max(10, int(self.N * 0.02))
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            cumulative_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(14, 7))
            plt.plot(cumulative_average, label='Cumulative Average Reward', linestyle='--', color='maroon')
            plt.plot(rolling_avg, label='Rolling Average Reward (window size={window_size})', color='darkblue')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('ε-greedy Hybrid: Reward Trends Over Time')
            plt.legend()
            plt.show()