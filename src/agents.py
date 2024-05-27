import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from sklearn.ensemble import RandomForestClassifier
from collections import namedtuple, deque
from tqdm import tqdm

#----------------------------------------------------------------------------------------------------------------------#
# Forest-Enhanced Dirichlet Sampling
# Applies Thompson Sampling to a Dirichlet distribution for multinomial reward modeling, based on papers below.
# Uses random forest predictions to boost parameters for genres likely to be favorites.
# - Riou and Honda, 2020: "Bandit Algorithms Based on Thompson Sampling for Bounded Reward Distributions"
# - Baudry, Saux, Maillard, 2021: "From Optimality to Robustness: Dirichlet Sampling Strategies in Stochastic Bandits"
#----------------------------------------------------------------------------------------------------------------------#

class DirichletForestSampling:
    def __init__(self, genre_list, steps, environment=None, forest=True):
        # K-armed bandit problem: genres as arms and user as environment
        self.k = len(genre_list)  
        self.N = steps  
        self.environment = environment
        self.arm_history = {arm_index: [] for arm_index in range(self.k)} 
        self.recent_rewards = []

        # Create a RandomForest instance, if requested
        self.forest = forest
        self.model = RandomForestClassifier(n_estimators=100) if forest else None 
        self.initialized = False
        
        # Uniform Dirichlet parameters as priors for each genre
        self.dirichlet_params = np.ones((self.k, 5))

    def run(self):
        rewards = []
        with tqdm(total=self.N, desc="{:40}".format("Current agent: Dirichlet Forest Sampling"), leave=False) as progress:
            for i in range(self.N):
                sampled_probs = [np.random.dirichlet(params) for params in self.dirichlet_params]  # Sample from Dirichlet...
                expected_rewards = [np.dot(probs, np.arange(1, 6)) for probs in sampled_probs]     # calculate EV...
                chosen_arm = np.argmax(expected_rewards)                                           # and chose arm w/ highest EV
                
                reward = self.environment.get_reward(chosen_arm)
                rewards.append(reward)
                self.arm_history[chosen_arm].append(reward)

                # Non-linear update to Dirichlet parameters, via quadratic fit to (1,1) and (5,2)
                increment = 1 + 0.1 * (reward - 1) + 0.0375 * (reward - 1)**2 
                self.dirichlet_params[chosen_arm][reward - 1] += increment

                # Periodically adapt Dirichlet parameters for well-performing genres
                if self.forest and (i + 1) % 1000 == 0:
                    self.train()
                    self.forest_boost()
                if (i + 1) % 100 == 0 and i >= 100:
                    self.adaptive_boost()
                progress.update(1)
                
        self.recent_rewards = rewards
        return rewards
    
    # Train the random forest classifier on accumulated history if sufficient data is available
    def train(self):
        X = []
        y = []
        for _, rewards in self.arm_history.items():
            if len(rewards) > 1:
                features = [                                        # Feature vector includes...
                    np.mean(rewards),                               # 1. average reward
                    np.std(rewards, ddof=1),                        # 2. standard deviation of rewards
                    np.sum(np.array(rewards) >= 4) / len(rewards),  # 3. proportion of high rewards (4 or 5)
                    len(rewards)                                    # 4. total number of rewards
                ]
                X.append(features)
                y.append(1 if np.mean(rewards) > 4 else 0)  # Same binary target as adaptive_boost
        if X:
            self.model.fit(X, y)
            self.initialized = True   # Update flag once model is trained

    # Boost Dirichlet parameters for genres predicted to be favorites by radom forest
    def forest_boost(self):
        for arm in range(self.k):
            if len(self.arm_history[arm]) > 1:
                features = [
                    np.mean(self.arm_history[arm]),
                    np.std(self.arm_history[arm], ddof=1),
                    np.sum(np.array(self.arm_history[arm]) >= 4) / len(self.arm_history[arm]),
                    len(self.arm_history[arm])
                ]
                if self.model.predict([features])[0] == 1:
                    self.dirichlet_params[arm] += np.array([0, 0, 0, 1, 2])

    # Boost the Dirichlet parameters based on the average rewards exceeding a threshold
    def adaptive_boost(self):
        average_rewards = [np.mean(history) if history else 0 for history in self.arm_history.values()]
        for i, avg in enumerate(average_rewards):
            if avg > 4:
                self.dirichlet_params[i] += np.array([0, 0, 1, 2, 3])

    # Clear history, reinitialize Dirichlet parameters to uniform priors, and reset classifier if needed
    def reset(self):
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.dirichlet_params = np.ones((self.k, 5))
        if self.forest:
            self.model = RandomForestClassifier(n_estimators=100)
            self.initialized = False

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
            plt.title('Dirichlet Forest Sampling: Learned Posterior Distribution by Genre')
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
            plt.title('Dirichlet Forest Sampling: Reward Trends Over Time')
            plt.legend()
            plt.show()

#---------------------------------------------------------------------------#
# DQN (Deep Q-Network)
# Utilizes neural networks to approximate the optimal action-value function.
#---------------------------------------------------------------------------#

# A single transition, mapping (state, action) pairs to their results
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

# Storage for transitions observed by agent, allowing for decorrelated training batches
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    # Save a transition
    def push(self, *args):
        self.memory.append(Transition(*args))

    # Randomly sample a batch of transitions
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DeepQNetwork:
    def __init__(self, genre_list, steps, environment=None, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=2000,
                 hidden_size=64, gamma=0.99, batch_size=256, target_update=400):
        self.environment = environment
        self.N = steps
        self.genre_list = genre_list                        # Actions: genres to choose from
        self.state_size = len(self.genre_list)
        self.action_size = len(self.genre_list)
        self.gamma = gamma                                  # Discount factor for future rewards 
        self.epsilon_start= epsilon_start                   # ε-decreasing: starting value...
        self.epsilon_final = epsilon_final                  # minimum value after decay...
        self.epsilon_decay = epsilon_decay                  # and steo when decay ends (controls decay rate)
        self.steps_done = 0
        self.batch_size = batch_size                        # Size of ReplayMemory batches (affects runtime)
        self.target_update = target_update                  # Frequency for training target network (affects runtime)
        self.memory = ReplayMemory(self.N)
        self.hidden_size = hidden_size
        self.policy_net = self.build_model(hidden_size)     # Policy network (predicts best actions for given state)
        self.target_net = self.build_model(hidden_size)     # Target network (stable baseline) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.recent_rewards = []

    # Construct NN with given hiden size
    def build_model(self, hidden_size):
        model = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)
        )
        return model

    # Select action using ε-decreasing policy
    def select_action(self, state):
        # Calculate current ε, with a gradual decay
        eps_threshold = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            # Exploit: current best arm
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Explore: random arm
            return torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long)

    # Interact w/ environment to collect rewards, training policy network on experiences
    def run(self):
        state = torch.zeros([1, self.state_size])   # No prior information
        rewards = []
        with tqdm(total=self.N, desc="{:40}".format("Current agent: Deep Q-Network"), leave=False) as progress:
            for step in range(self.N):
                action = self.select_action(state)                      # Choose an action...
                reward = self.environment.get_reward(action.item())     # collect the reward...
                rewards.append(reward)                                  # and store it
                next_state = torch.zeros([1, self.state_size])
                self.memory.push(state, action, next_state, torch.tensor([reward], dtype=torch.float32))
                state = next_state
                self.optimize_model()

                # Periodically set target network weights equal to policy network weights
                if step % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                progress.update(1)

        self.recent_rewards = rewards
        return rewards

    # Optimize policy network w/ minibatch of experiences from memory, using Huber loss to learn Q values
    def optimize_model(self):
        if len(self.memory) < self.batch_size:              # If memory has enough experiences...
            return
        transitions = self.memory.sample(self.batch_size)   # sample a batch of transitions...
        batch = Transition(*zip(*transitions))              # and reform batch data

        # Filter out non-final states and construct tensors
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute current Q values w/ policy network
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values w/ target network, and expected Q values based on rewards
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss, update weights
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    # Reset agent to start a new episode or training session
    def reset(self):
        self.steps_done = 0  
        self.epsilon = self.epsilon_start  
        self.memory = ReplayMemory(10000)   # Optionally clear memory
        
        # Uncomment to reinitialize network weights, comment out to continue with learned weights
        self.policy_net = self.build_model(self.hidden_size)  
        self.target_net = self.build_model(self.hidden_size) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        
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
            plt.title('Deep Q-Network: Learned Posterior Distribution by Genre')
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
            plt.title('Deep Q-Network: Reward Trends Over Time')
            plt.legend()
            plt.show()

#----------------------------------------------------------------#
# LinUCB (upper confidence bound)
# Uses linear models with upper confidence bounds for decisions.
#----------------------------------------------------------------#

class LinUCB:
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
        with tqdm(total=self.N, desc="{:40}".format("Current agent: LinUCB"), leave=False) as progress:
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
                progress.update(1)
            
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
        
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
        if plot_option in ["both", "posterior"]:
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
            
        # 2. Plot cumulative average reeward and sliding window average
        if plot_option in ["both", "trends"]:
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
            
#-----------------------------------------------------------------------------------------------#
# Advantage Actor-Critic (A2C)
# Uses separate network for policy (actor) and value (critic) estimation to stabilize training.
#-----------------------------------------------------------------------------------------------#
 
class AdvantageActorCritic:
    def __init__(self, genre_list, steps, environment, hidden_size=128, gamma=0.99, lr=1e-3, entropy_coef=0.01):
        self.environment = environment
        self.N = steps
        self.genre_list = genre_list
        self.action_size = len(genre_list)      # Actions: genres to choose from
        self.recent_rewards = []
        
        # State includes counts and cumulative rewards for each genre
        self.state_size = 2 * len(genre_list)  # Two features per genre
        self.gamma = gamma                     # Discount factor for future rewards 
        self.lr = lr                           # Learning rate 
        self.entropy_coef = entropy_coef       # Randomness in policy, encouraging exploration
        self.hidden_size = hidden_size

        # Actor and critic networks
        self.actor = self.build_model(self.hidden_size, self.action_size)
        self.critic = self.build_model(self.hidden_size, 1)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        
        # Initialize state with zeros
        self.action_counts = np.zeros(len(genre_list))
        self.cumulative_rewards = np.zeros(len(genre_list))
        self.state = np.concatenate([self.action_counts, self.cumulative_rewards])

    # Construct NN with given hiden size
    def build_model(self, hidden_size, output_size):
        model = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        return model

    # Convert state to tensor and get action probabilities from actor network
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        probs = torch.softmax(self.actor(state), dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    # Update action counts and cumulative rewards for chosen action
    def update_state(self, action, reward):
        # Update action counts and cumulative rewards
        self.action_counts[action] += 1             
        self.cumulative_rewards[action] += reward 
        
        # Normalize features to prevent scale issues
        normalized_counts = self.action_counts / np.sum(self.action_counts)  
        max_reward = np.max(self.cumulative_rewards) if np.max(self.cumulative_rewards) != 0 else 1
        normalized_rewards = self.cumulative_rewards / max_reward
        self.state = np.concatenate([normalized_counts, normalized_rewards])

    # Compute advantage from reward and value estimation from critic
    def optimize_model(self, log_prob, reward, value):
        reward = torch.tensor([reward], dtype=torch.float)    # Convert reward to tensor if it's not already
        advantage = reward - value
        actor_loss = -log_prob * advantage                    # Actor loss (negative log likelihood * advantage)
        critic_loss = advantage.pow(2)                        # Critic loss (mean squared value error)
        entropy = -torch.sum(torch.exp(log_prob) * log_prob)  # Entropy bonus for exploration

        # Combine losses for backpropagation
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy  
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def run(self):
        rewards = []
        with tqdm(total=self.N, desc="{:40}".format("Current agent: Advantage Actor-Critic"), leave=False) as progress: 
            for _ in range(self.N):
                # Select an action based on the current state and receive a reward
                action, log_prob = self.select_action(self.state)
                reward = self.environment.get_reward(action)
                rewards.append(reward)
                
                # Value estimate from critic for current state
                value = self.critic(torch.tensor(self.state, dtype=torch.float).unsqueeze(0))
                
                # Update the state and optimize both actor and critic networks
                self.update_state(action, reward)
                self.optimize_model(log_prob, reward, value)
                progress.update(1)
                
        self.recent_rewards = rewards
        return rewards
    
    # Reset function to reinitialize variables for a new run or episode
    def reset(self):
        self.action_counts = np.zeros(len(self.genre_list))
        self.cumulative_rewards = np.zeros(len(self.genre_list))
        self.state = np.concatenate([self.action_counts, self.cumulative_rewards])
        
        # Rebuild models using the saved hidden_size
        self.actor = self.build_model(self.hidden_size, self.action_size)
        self.critic = self.build_model(self.hidden_size, 1)
        
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
            plt.title('Advantage Actor-Critic: Learned Posterior Distribution by Genre')
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
            plt.title('Advantage Actor-Critic: Reward Trends Over Time')
            plt.legend()
            plt.show()

#-------------------------------------------------------------------------------#
# ε-decreasing 
# Balances exploitation and exploration; adaptively decreases exploration rate.
#-------------------------------------------------------------------------------#

class EpsilonDecreasing:
    def __init__(self, genre_list, steps, environment=None):
        self.genre_list = genre_list
        
        # K-armed bandit problem: genres as arms and user as environment
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
        with tqdm(total=self.N, desc="{:40}".format("Current agent: ε-decreasing"), leave=False) as progress:
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
                
                # Update the arm's expected value (EV) and history
                reward = self.environment.get_reward(arm)
                rewards.append(reward)
                n = self.arm_counts[arm] + 1
                self.arm_counts[arm] = n
                current_EV = self.arm_EV[arm]
                new_EV = current_EV + (reward - current_EV) / n
                self.arm_EV[arm] = new_EV
                self.arm_history[arm].append(reward)
                progress.update(1)
            
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
            
#---------------------------------------------------------------------------#
# ε-decreasing hybrid
# Mixes Dirichlet Sampling and UCB strategies based on decaying probability.
#---------------------------------------------------------------------------#

class EpsilonDecreasingHybrid:
    def __init__(self, genre_list, steps, epsilon=0.276, environment=None):
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
        with tqdm(total=self.N, desc="{:40}".format("Current agent: ε-decreasing hybrid"), leave=False) as progress:
            for i in range(self.N):
                # Dynamic exploration factor that decreases as the number of steps increases
                exploration_prob = self.epsilon * np.log(1 + i) / np.log(1 + self.N)
                if np.random.rand() < exploration_prob:
                    # Use UCB based on the dynamic exploration factor...
                    confidence_bounds = [np.mean(self.dirichlet_params[arm]) + np.sqrt(2 * np.log(i+1) / (self.arm_counts[arm] + 1e-10)) for arm in range(self.k)]
                    chosen_arm = np.argmax(confidence_bounds)
                else:
                    # otherwise use Dirichlet Sampling
                    sampled_probs = [np.random.dirichlet(params) for params in self.dirichlet_params] 
                    expected_rewards = [np.dot(probs, np.arange(1, 6)) for probs in sampled_probs]
                    chosen_arm = np.argmax(expected_rewards)
                    
                # Log rewards in history and update Dirichlet parameters
                reward = self.environment.get_reward(chosen_arm)
                rewards.append(reward)
                self.recent_rewards.append(reward)
                self.arm_history[chosen_arm].append(reward)
                self.dirichlet_params[chosen_arm][reward - 1] += 1
                self.arm_counts[chosen_arm] += 1
                progress.update(1)
                
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

#---------------------------------------------------------------------#
# ε-first
# Explores uniformly early on, then exclusively exploits best option.
#---------------------------------------------------------------------#

class EpsilonFirst:
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
        best_arm = None  # Initialize best_arm
        with tqdm(total=self.N, desc="{:40}".format("Current agent: ε-first"), leave=False) as progress:
            for i in range(self.N):
                if i < (self.epsilon * self.N):  # Exploration phase: select a random arm
                    arm = np.random.randint(0, self.k)
                    reward = self.environment.get_reward(arm)
                    self.arm_history[arm].append(reward)
                    
                elif i == int(self.epsilon * self.N):  # Decision point: find arm with highest expected value
                    means = np.array([np.mean(history) if history else 0 for history in self.arm_history.values()])
                    best_arm = np.argmax(means)
                    reward = self.environment.get_reward(best_arm)
                    
                else:  # Exploitation phase: repeatedly select arm with highest expected value
                    reward = self.environment.get_reward(best_arm)
                    
                rewards.append(reward)
                progress.update(1)
                
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
        
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
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
        
        # 2. Plot cumulative average reeward and sliding window average
        if plot_option in ["both", "trends"]:
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

class EpsilonGreedy:
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
        with tqdm(total=self.N, desc="{:40}".format("Current agent: ε-greedy"), leave=False) as progress:
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
                progress.update(1)
            
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
        
        # 1. Plot learned posteriors (Using KDE for a smooth curve)
        if plot_option in ["both", "posterior"]:
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
            plt.title('ε-greedy: Reward Trends Over Time')
            plt.legend()
            plt.show()

#------------------------------------------------------------------------#
# ε-greedy hybrid
# Mixes Dirichlet Sampling and UCB strategies based on fixed probability.
#------------------------------------------------------------------------#
class EpsilonGreedyHybrid:
    def __init__(self, genre_list, steps, epsilon=0.022, environment=None):
        self.k = len(genre_list)
        
        # K-armed bandit problem: genres as arms and user as environment
        self.N = steps
        self.epsilon = epsilon
        self.environment = environment
        self.dirichlet_params = np.ones((self.k, 5))
        self.recent_rewards = []
        self.arm_history = {arm_index: [] for arm_index in range(self.k)}
        self.arm_counts = np.zeros(self.k)

    # At each step, ε probability to select LinUCB, 1-ε probability to select Dirichlet sampling
    def run(self):
        rewards = []
        with tqdm(total=self.N, desc="{:40}".format("Current agent: ε-greedy Hybrid"), leave=False) as progress:
            for i in range(self.N):
                if np.random.rand() < self.epsilon:
                    # LinUCB: select arm with highest predicted reward + confidence,
                    confidence_bounds = [np.mean(self.dirichlet_params[arm]) + np.sqrt(2 * np.log(i+1) / (self.arm_counts[arm] + 1e-10)) for arm in range(self.k)]
                    chosen_arm = np.argmax(confidence_bounds)
                else:
                    # Dirichlet Sampling: sample a probability vector to calculate EV, then select the highest EV arm
                    sampled_probs = [np.random.dirichlet(params) for params in self.dirichlet_params] 
                    expected_rewards = [np.dot(probs, np.arange(1, 6)) for probs in sampled_probs]
                    chosen_arm = np.argmax(expected_rewards)
                
                # Log rewards in history and update Dirichlet parameters
                reward = self.environment.get_reward(chosen_arm)
                rewards.append(reward)
                self.recent_rewards.append(reward)
                self.arm_history[chosen_arm].append(reward)
                self.dirichlet_params[chosen_arm][reward - 1] += 1
                self.arm_counts[chosen_arm] += 1
                progress.update(1)
                
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

#------------------------------------------------------------------#
# A/B testing
# Transitions from alpha test phase to perpetual beta test phase.
#------------------------------------------------------------------#

class ABTesting:
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
        with tqdm(total=self.N, desc="{:40}".format("Current agent: A/B Testing"), leave=False) as progress:
    
            # Alpha phase: initial uniform exploration for 20% of total steps
            for t in range(exploration_steps):
                chosen_arm = t % self.k
                reward = self.environment.get_reward(chosen_arm)
                rewards.append(reward)
                self.arm_history[chosen_arm].append(reward)
                self.arm_rewards[chosen_arm] += reward
                self.arm_counts[chosen_arm] += 1
                progress.update(1) 

            # Perpetual beta phase: Exploit the best arms for 80%, continue to explore for 20%
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
                progress.update(1) 

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