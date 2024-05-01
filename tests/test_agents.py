import pytest
import numpy as np
import torch
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
    from src.agents import *
except ModuleNotFoundError:
    from agents import *

#--------------------------------------------------#
# Static enviornment to evaluate decision policies
#--------------------------------------------------#

# A simple stationary enviornment (always returns fixed reward for each genre)
class DummyEnvironment:
    def __init__(self, rewards):
        self.rewards = rewards

    # Return a predetermined reward based on arm index (fixed)
    def get_reward(self, arm_index):
        return self.rewards[arm_index]

#-----------------------------------------------------------------------------------------------#
# Test Suite for Agents
# Bandit algorithms: verify initialization, balance of exploration/exploitation, and resetting.
# Deep RL: additionally validate action selection, learning processes, and memory management. 
#-----------------------------------------------------------------------------------------------#

class TestDirichletForestSampling:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1, 2, 5]

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def dfs_agent(self, genre_list, environment):
        # Initialize the agent with the option to use a RandomForest
        return DirichletForestSampling(genre_list, 1000, environment, forest=True)

    # Verify that DirichletForestSampling agent initializes correctly with expected parameters
    def test_initialization(self, dfs_agent):
        assert isinstance(dfs_agent.model, RandomForestClassifier), "Model should be initialized when forest is True."
        assert dfs_agent.forest, "Forest flag should be True when RandomForest is used."

    # Verify that agent can interact with environemnt throughout a run
    def test_run_functionality(self, dfs_agent):
        rewards = dfs_agent.run()
        assert len(rewards) == dfs_agent.N, "Should run for N steps and collect N rewards."
        assert all(isinstance(reward, int) for reward in rewards), "All rewards should be integers."

    # Verify that RandomForest identifies parameters to boost
    def test_model_training_and_boosting(self, dfs_agent):
        # Populate some history to ensure there's enough data to train the model
        for _ in range(100):
            dfs_agent.arm_history[0].append(5)
        dfs_agent.train()
        assert dfs_agent.initialized, "Model should be trained and initialized flag set."
        
        initial_params = dfs_agent.dirichlet_params[0].copy()
        dfs_agent.forest_boost()
        assert np.any(dfs_agent.dirichlet_params[0] > initial_params), "Parameters should be boosted based on model predictions."

    # Verify that reset properly clears agent's state
    def test_reset_functionality(self, dfs_agent):
        dfs_agent.run()
        dfs_agent.reset()
        assert all(np.array_equal(params, np.ones(5)) for params in dfs_agent.dirichlet_params), "Parameters should be reset to initial uniform values."
        assert len(dfs_agent.recent_rewards) == 0, "Recent rewards should be cleared on reset."
        assert all(len(hist) == 0 for hist in dfs_agent.arm_history.values()), "Arm history should be cleared on reset."

class TestDeepQNetwork:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1.0, 2.0, 3.0]

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def dqn_agent(self, genre_list, environment):
        agent = DeepQNetwork(genre_list, 1000, environment)
        # Prepopulate the memory to ensure there's enough data for training
        for _ in range(agent.batch_size + 1):
            state = torch.zeros([1, agent.state_size])
            action = agent.select_action(state)
            reward = agent.environment.get_reward(action.item())
            next_state = torch.zeros([1, agent.state_size])
            agent.memory.push(state, torch.tensor([[action]], dtype=torch.long), next_state, torch.tensor([reward], dtype=torch.float32))
        return agent

    # Verify that DeepQNetwork agent initializes correctly with expected parameters
    def test_initialization(self, dqn_agent):
        assert dqn_agent.genre_list is not None, "Genre list should be initialized."
        assert dqn_agent.N == 1000, "Total number of steps should be correctly set."

    def test_action_selection(self, dqn_agent):
        state = torch.zeros([1, dqn_agent.state_size])
        action = dqn_agent.select_action(state)
        assert action is not None, "Action should be selected."

    # Test optimization (memory is populated, so batch size requirement is met)
    def test_learning_process(self, dqn_agent):
        initial_optim_state = [param.clone() for param in dqn_agent.policy_net.parameters()]
        dqn_agent.optimize_model() 
        updated_optim_state = [param for param in dqn_agent.policy_net.parameters()]
        assert any(not torch.equal(i, u) for i, u in zip(initial_optim_state, updated_optim_state)), "Weights should update after learning."

    # Verify that reset properly clears agent's state
    def test_reset(self, dqn_agent):
        dqn_agent.run()
        dqn_agent.reset()
        assert dqn_agent.steps_done == 0, "Steps done should be reset to zero."
        assert len(dqn_agent.memory) == 0, "Memory should be cleared on reset."

class TestLinUCB:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1.0, 2.0, 3.0]

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def lin_ucb_agent(self, genre_list, environment):
        return LinUCB(genre_list, 1000, environment)

    # Verify that LinUCB agent initializes correctly with expected parameters
    def test_initialization(self, lin_ucb_agent):
        assert lin_ucb_agent.genre_list is not None, "Genre list should be initialized."
        assert len(lin_ucb_agent.genre_list) == lin_ucb_agent.k, "Number of arms should match number of genres."
        assert lin_ucb_agent.N == 1000, "Total number of steps should be correctly set."
        assert all(a == 0 for a in lin_ucb_agent.arm_counts), "Arm counts should initially be zero."
        assert all(ev == 0 for ev in lin_ucb_agent.arm_EV), "Expected values should initially be zero."

    # Test the core functionality of LinUCB algorithm
    def test_exploration_and_exploitation(self, lin_ucb_agent):
        lin_ucb_agent.run()
        assert all(count > 0 for count in lin_ucb_agent.arm_counts), "Each arm should be explored at least once."
        assert max(lin_ucb_agent.arm_EV) == max(lin_ucb_agent.recent_rewards), "The arm with the highest reward should have the highest EV."

    # Ensure that reset properly clears the agent's state
    def test_reset(self, lin_ucb_agent):
        lin_ucb_agent.run()
        lin_ucb_agent.reset()
        assert all(ev == 0 for ev in lin_ucb_agent.arm_EV), "Expected values should be reset to zero."
        assert all(count == 0 for count in lin_ucb_agent.arm_counts), "Arm counts should be reset to zero."
        assert lin_ucb_agent.recent_rewards == [], "Recent rewards list should be cleared."
        assert all(len(history) == 0 for history in lin_ucb_agent.arm_history.values()), "Arm history should be cleared upon reset."

class TestAdvantageActorCritic:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1.0, 2.0, 3.0]

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def advantage_actor_critic_agent(self, genre_list, environment):
        return AdvantageActorCritic(genre_list, 1000, environment)

    # Verify that AdvantageActorCritic agent initializes correctly with expected parameters
    def test_initialization(self, advantage_actor_critic_agent):
        assert advantage_actor_critic_agent.genre_list is not None, "Genre list should be initialized."
        assert advantage_actor_critic_agent.N == 1000, "Total number of steps should be correctly set."
        assert all(a == 0 for a in advantage_actor_critic_agent.action_counts), "Action counts should initially be zero."
        assert all(r == 0 for r in advantage_actor_critic_agent.cumulative_rewards), "Cumulative rewards should initially be zero."

    # Verify actions are selected and states are updated correctly
    def test_action_selection_and_state_update(self, advantage_actor_critic_agent):
        initial_state = advantage_actor_critic_agent.state.copy()
        action, log_prob = advantage_actor_critic_agent.select_action(initial_state)
        assert isinstance(action, int) and action in range(len(advantage_actor_critic_agent.genre_list)), "Selected action should be valid."
        reward = advantage_actor_critic_agent.environment.get_reward(action)
        advantage_actor_critic_agent.update_state(action, reward)
        assert not np.array_equal(initial_state, advantage_actor_critic_agent.state), "State should be updated after action."

    # Test forward and backward pass of training
    def test_network_training(self, advantage_actor_critic_agent):
        action, log_prob = advantage_actor_critic_agent.select_action(advantage_actor_critic_agent.state)
        reward = advantage_actor_critic_agent.environment.get_reward(action)
        value = advantage_actor_critic_agent.critic(torch.tensor(advantage_actor_critic_agent.state, dtype=torch.float).unsqueeze(0))
        old_actor_params = [param.clone() for param in advantage_actor_critic_agent.actor.parameters()]
        old_critic_params = [param.clone() for param in advantage_actor_critic_agent.critic.parameters()]
        advantage_actor_critic_agent.optimize_model(log_prob, reward, value)
        new_actor_params = [param for param in advantage_actor_critic_agent.actor.parameters()]
        new_critic_params = [param for param in advantage_actor_critic_agent.critic.parameters()]

        # Ensure parameters have been updated (indicating training has occurred)
        assert any(not torch.equal(o, n) for o, n in zip(old_actor_params, new_actor_params)), "Actor parameters should be updated after training."
        assert any(not torch.equal(o, n) for o, n in zip(old_critic_params, new_critic_params)), "Critic parameters should be updated after training."

    # Verify reinitialization of internal state variables and neural network models to their initial state
    def test_reset(self, advantage_actor_critic_agent):
        advantage_actor_critic_agent.run()
        advantage_actor_critic_agent.reset()
        assert all(a == 0 for a in advantage_actor_critic_agent.action_counts), "Action counts should be reset to zero."
        assert all(r == 0 for r in advantage_actor_critic_agent.cumulative_rewards), "Cumulative rewards should be reset to zero."
        assert np.allclose(advantage_actor_critic_agent.state, np.zeros_like(advantage_actor_critic_agent.state)), "State should be reset to initial condition."
        assert advantage_actor_critic_agent.actor is not None, "Actor network should be reinitialized."
        assert advantage_actor_critic_agent.critic is not None, "Critic network should be reinitialized."

class TestEpsilonDecreasing:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1.0, 2.0, 3.0]  # Fixed rewards for simplicity

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def epsilon_decreasing_agent(self, genre_list, environment):
        return EpsilonDecreasing(genre_list, 1000, environment)

    # Test that the EpsilonDecreasing agent initializes correctly with expected parameters
    def test_initialization(self, epsilon_decreasing_agent):
        assert epsilon_decreasing_agent.genre_list is not None, "Genre list should be initialized."
        assert epsilon_decreasing_agent.k == len(epsilon_decreasing_agent.genre_list), "Number of arms should match the number of genres."
        assert epsilon_decreasing_agent.N == 1000, "Total number of steps should be correctly set."
        assert epsilon_decreasing_agent.initial_epsilon == 0.99, "Initial epsilon should be set to 0.99."
        assert epsilon_decreasing_agent.minimum_epsilon == 0.01, "Minimum epsilon should be set to 0.01."
        assert epsilon_decreasing_agent.recent_rewards == [], "Recent rewards should start as an empty list."

    # Verify that the reset function correctly clears the agent's internal state
    def test_reset(self, epsilon_decreasing_agent):
        epsilon_decreasing_agent.run()
        epsilon_decreasing_agent.reset()
        assert all(ev == 0 for ev in epsilon_decreasing_agent.arm_EV), "All EVs should be reset to zero."
        assert all(count == 0 for count in epsilon_decreasing_agent.arm_counts), "All arm counts should be reset to zero."
        assert epsilon_decreasing_agent.recent_rewards == [], "Recent rewards should be cleared upon reset."

    # Verify that agent's learned best arm matches the theoretical best arm
    def test_convergence_to_best_arm(self, epsilon_decreasing_agent):
        epsilon_decreasing_agent.run()
        best_arm = np.argmax(epsilon_decreasing_agent.arm_EV)
        theoretical_best_arm = np.argmax(epsilon_decreasing_agent.environment.rewards)
        assert best_arm == theoretical_best_arm, "The agent should converge to the best arm."

class TestEpsilonFirst:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1.0, 2.0, 3.0]  

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def epsilon_first_agent(self, genre_list, environment):
        return EpsilonFirst(genre_list, 1000, epsilon=0.1, environment=environment)

    # Verify that EpsilonFirst agent initializes correctly with expected parameters
    def test_initialization(self, epsilon_first_agent):
        assert epsilon_first_agent.genre_list is not None, "Genre list should be initialized."
        assert epsilon_first_agent.k == len(epsilon_first_agent.genre_list), "Number of arms should match the number of genres."
        assert epsilon_first_agent.N == 1000, "Total number of steps should be correctly set."
        assert epsilon_first_agent.epsilon == 0.1, "Epsilon should be set to 0.1."
        assert epsilon_first_agent.recent_rewards == [], "Recent rewards should start as an empty list."

    # Verify that reset function correctly clears agent's internal state
    def test_reset(self, epsilon_first_agent):
        epsilon_first_agent.run()
        epsilon_first_agent.reset()
        assert epsilon_first_agent.recent_rewards == [], "Recent rewards should be cleared upon reset."
        assert all(len(history) == 0 for history in epsilon_first_agent.arm_history.values()), "Arm history should be cleared upon reset."

    # Compare best arm from exploration against arms chosen during explotation
    def test_effective_strategy(self, epsilon_first_agent):
        epsilon_first_agent.run()
        exploration_phase_length = int(epsilon_first_agent.epsilon * epsilon_first_agent.N)
        means = [np.mean(history) if history else 0 for history in epsilon_first_agent.arm_history.values()]
        best_arm_during_exploration = np.argmax(means)
        expected_best_reward = np.mean(epsilon_first_agent.arm_history[best_arm_during_exploration])
        exploitation_rewards = epsilon_first_agent.recent_rewards[exploration_phase_length:]

        # Check if mean of exploitation rewards closely matches expected best reward
        if len(exploitation_rewards) > 0: 
            actual_mean_exploitation_reward = np.mean(exploitation_rewards)
            assert actual_mean_exploitation_reward == expected_best_reward, "The mean reward during exploitation should closely match the mean reward of the best arm found during exploration."
        else:
            pytest.fail("No exploitation actions found. Check the epsilon value and total steps.")


class TestEpsilonGreedy:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1.0, 2.0, 3.0]

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def epsilon_greedy_agent(self, genre_list, environment):
        return EpsilonGreedy(genre_list, 1000, epsilon=0.1, environment=environment)

    # Verify that EpsilonGreedy agent initializes correctly with expected parameters
    def test_initialization(self, epsilon_greedy_agent):
        assert epsilon_greedy_agent.genre_list is not None, "Genre list should be initialized."
        assert epsilon_greedy_agent.k == len(epsilon_greedy_agent.genre_list), "Number of arms should match the number of genres."
        assert epsilon_greedy_agent.N == 1000, "Total number of steps should be correctly set."
        assert epsilon_greedy_agent.epsilon == 0.1, "Epsilon should be set to 0.1."
        assert len(epsilon_greedy_agent.arm_EV) == epsilon_greedy_agent.k, "Expected values should be initialized to zero for each genre."
        assert len(epsilon_greedy_agent.arm_counts) == epsilon_greedy_agent.k, "Arm counts should be initialized to zero for each genre."
        assert epsilon_greedy_agent.recent_rewards == [], "Recent rewards should start as an empty list."

    # Verify that reset function correctly clears agent's internal state
    def test_reset(self, epsilon_greedy_agent):
        epsilon_greedy_agent.run()
        epsilon_greedy_agent.reset()
        assert all(ev == 0 for ev in epsilon_greedy_agent.arm_EV), "All EVs should be reset to zero."
        assert all(count == 0 for count in epsilon_greedy_agent.arm_counts), "All arm counts should be reset to zero."
        assert epsilon_greedy_agent.recent_rewards == [], "Recent rewards should be cleared upon reset."

    # Verify that agent effectively learns to prefer best performing arm over time
    def test_effective_strategy(self, epsilon_greedy_agent):
        epsilon_greedy_agent.run()
        expected_best_arm = np.argmax(epsilon_greedy_agent.environment.rewards)  
        most_chosen_arm = np.argmax(epsilon_greedy_agent.arm_counts) 
        assert most_chosen_arm == expected_best_arm, "The most chosen arm should be the one with the highest rewards."

    # Test balance between exploration and exploitation
    def test_exploration_exploitation_balance(self, epsilon_greedy_agent):
        epsilon_greedy_agent.run()
        total_explorations = sum(1 for i in epsilon_greedy_agent.recent_rewards if np.random.random() < epsilon_greedy_agent.epsilon)
        assert total_explorations / epsilon_greedy_agent.N <= (epsilon_greedy_agent.epsilon + 0.05), "Exploration rate should be within expected bounds."

class TestABTesting:
    @pytest.fixture
    def genre_list(self):
        return ['Action', 'Comedy', 'Drama']

    @pytest.fixture
    def rewards(self):
        return [1.0, 2.0, 3.0]  

    @pytest.fixture
    def environment(self, rewards):
        return DummyEnvironment(rewards)

    @pytest.fixture
    def ab_testing_agent(self, genre_list, environment):
        return ABTesting(genre_list, 1000, environment)

    # Verifiy that ABTesting agent initializes correctly with expected parameters
    def test_initialization(self, ab_testing_agent):
        assert ab_testing_agent.genre_list is not None, "Genre list should be initialized."
        assert ab_testing_agent.k == len(ab_testing_agent.genre_list), "Number of arms should match the number of genres."
        assert ab_testing_agent.N == 1000, "Total number of steps should be correctly set."
        assert len(ab_testing_agent.arm_rewards) == ab_testing_agent.k, "Arm rewards should be initialized to zero for each genre."
        assert len(ab_testing_agent.arm_counts) == ab_testing_agent.k, "Arm counts should be initialized to zero for each genre."
        assert ab_testing_agent.recent_rewards == [], "Recent rewards should start as an empty list."

    # Verify that reset function correctly clears agent's internal state
    def test_reset(self, ab_testing_agent):
        ab_testing_agent.run()
        ab_testing_agent.reset()
        assert all(reward == 0 for reward in ab_testing_agent.arm_rewards), "All arm rewards should be reset to zero."
        assert all(count == 0 for count in ab_testing_agent.arm_counts), "All arm counts should be reset to zero."
        assert ab_testing_agent.recent_rewards == [], "Recent rewards should be cleared upon reset."

    # Verify that agent effectively learns to prefer best performing arm over time
    def test_effective_strategy(self, ab_testing_agent):
        ab_testing_agent.run()
        expected_best_arm = np.argmax(ab_testing_agent.environment.rewards)  
        most_chosen_arm = np.argmax(ab_testing_agent.arm_counts) 
        assert most_chosen_arm == expected_best_arm, "The most chosen arm should be the one with the highest rewards."

if __name__ == "__main__":
    pytest.main([__file__])