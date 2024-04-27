import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environments import *  
from agents import *        
from data_loader import *

#-----------------------#
# Simulation parameters
#-----------------------#

cmap = plt.get_cmap('nipy_spectral') # nipy_spectral isn't bad...
#np.random.seed(77)
np.random.seed(100)
N = 10000
total_users = 100
dynamic_epsilon = 0.276
hybrid_epsilon = 0.022

#----------------------------------------------------#
# Enviornment helper functions: setup and simulation
#----------------------------------------------------#

# Create environment, checking if environment class requires niche genres 
def create(env_class, genres, distributions, niche_genres, index):
    if env_class in [MultipleNicheGenreLoyalistEnvironment, NicheGenreLoyalistEnvironment]:
        return env_class(genres, distributions, niche_genres, index)
    else:
        return env_class(genres, distributions, index)

# Run agents on the specified environment
def run(environment, agents, steps=N):
    results = {}
    # Create a set of agents for the current environment, with flexibility in ε 
    for agent_name, agent_class in agents.items():
        if agent_name == "ε-Decreasing Hybrid":
            agent = agent_class(genres, steps, epsilon=dynamic_epsilon, environment=environment)
        elif agent_name == "ε-Greedy Hybrid":
            agent = agent_class(genres, steps, epsilon=hybrid_epsilon, environment=environment)
        else:
            agent = agent_class(genres, steps, environment=environment)
        rewards = agent.run()
        results[agent_name] = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)  # Compute and store cumulative average
    return results

#--------------------------#
# Load and preprocess data
#--------------------------#

data_path = "/Users/kahaan/Desktop/multi-armed-bandits/data/"
subset = load_subset(data_path)
genres, unnormalized_distributions, niche_genres = preprocess(subset, verbose=False)

#--------------------------------#
# Define environments and agents
#--------------------------------#

# Use proportions from mining_user_profiles.ipynb to construct an ensemble of users
environments = [
    (MultipleNicheGenreLoyalistEnvironment, int(0.22 * total_users)),
    (MultipleGenreEnjoyerEnvironment, int(0.44 * total_users)),
    (GenreEnjoyerEnvironment, int(0.06 * total_users)),
    (NicheGenreLoyalistEnvironment, int(0.10 * total_users)),
    (AverageViewerEnvironment, int(0.18 * total_users)),
]

agents = {
    'Dirichlet Sampling': DirichletSamplingAgent,
    'ε-decreasing Hybrid': EpsilonDecreasingHybridAgent,
    'ε-greedy Hybrid': EpsilonGreedyHybridAgent,
    'ε-first': EpsilonFirstAgent,
    'ε-greedy': EpsilonGreedyAgent,
    'LinUCB': LinUCBAgent,
    'ε-decreasing': EpsilonDecreasingAgent,
    'A/B Testing': ABTestingAgent,
}

#----------------#
# Run simulation
#----------------#

# Set up CLI progress bar
overall_results = {name: [] for name in agents}
total_environments = sum(count for _, count in environments) 
progress_bar = tqdm(total=total_environments, desc="Running Simulations")  

# Create each requested enviornment, then run agents on it and record the results
for env_class, count in environments:
    for i in range(count):
        env = create(env_class, genres, unnormalized_distributions, niche_genres, i+1)
        results = run(env, agents)
        for name in agents:
            overall_results[name].append(results[name])
        progress_bar.update(1)  
progress_bar.close()

#-------------------------------#
# Evaluating agent performances
#-------------------------------#

# Plot settings
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.grid'] = True
colors = [cmap(i) for i in np.linspace(0, 1, len(agents))]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

aggregated_results = {name: np.mean(overall_results[name], axis=0) for name in agents}
window_size = int(N * 0.0025)

# Plot rolling window average with markers...
for index, (name, results) in enumerate(aggregated_results.items()):
    rolling_avg = np.convolve(results, np.ones(window_size)/window_size, mode='valid')
    axes[0].plot(rolling_avg, label=name, color=colors[index], marker='^', markersize=4.5, markevery=500)
axes[0].set_title('Rolling Window Reward')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Average Rating')
axes[0].legend()

# and cumulative average plots with markers
end_values = []
for index, (name, results) in enumerate(aggregated_results.items()):
    cumulative_avg = np.cumsum(results) / np.arange(1, len(results) + 1)
    axes[1].plot(cumulative_avg, label=name, color=colors[index], marker='^', markersize=4.5, markevery=500)
    # Collecting the last value of each agent for ranking purposes
    end_values.append((name, cumulative_avg[-1]))
axes[1].set_title('Cumulative Reward')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Average Rating')
axes[1].legend()
plt.show()


# Display final rankings based on the cumulative averages (~EV)
end_values.sort(key=lambda x: x[1], reverse=True)
rankings = "\n".join([f"{idx + 1}. {name} - {value:.2f}" for idx, (name, value) in enumerate(end_values)])
print("Rankings based on the final values of the cumulative averages:")
print(rankings)