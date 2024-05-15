## Multi-Armed Bandit Problems

The Multi-Armed Bandit (MAB) problem, a one-state Markov decision process, is a powerful framework in decision theory and reinforcement learning. It describes a scenario where an agent, faced with multiple options (referred to as "arms"), must repeatedly choose among them in a series of trials to maximize cumulative reward. While the problem has been formulated in various ways, I focus on stochastic and stationary reward distributions, where each arm provides a random reward drawn from a distribution that remains constant over time. The agent's goal is to identify the arm that yields the highest expected reward. This involves devising a strategy that balances exploration (trying different arms to gather information) and exploitation (choosing the best-known arm to maximize immediate reward). Too much exploration reduces potential rewards by wasting time on suboptimal arms, while too much exploitation can lead to missing out on better options -- optimizing total reward requires balancing these strategies.

Formalizing the $K$-armed bandit problem, let $K \in \mathbb{N}^+$ be the number of arms, $T \in \mathbb{N}^+$ be the number of turns, $A_t \in \{1,2,...,K\}$ be the action chosen at time $t$, and $R_{t,A_t}$ be the reward recieved at time $t$ for choosing action $A_t$, drawn from an unknown stationary distribution. At each time step $t$, the agent selects an action $A_t$ from the set of arms, and the chosen arm provides a reward $R_{t, A_t}$. The process is repeated for $T$ trials, and the agent's objective is to maximize the expected cumulative reward $\mathbb{E}\left[  \sum_{t=1}^T R_{t, A_t} \right]$. 

MAB strategies are widely used across industries to optimize decisions under uncertainty and address the cold start problem -— where traditional machine learning and collaborative filtering methods struggle due to insufficient data or context for accurate predictions. In content recommendation systems like those used by Spotify, MAB algorithms dynamically select genres and songs based on real-time user interactions, quickly adapting to users [1]. In online advertising, companies like Facebook use MAB algorithms to optimize ad selection with limited initial data [2]. MAB algorithms also help Yahoo test headlines for user engagement [3].

## Existing Bandit Algorithms

There are many existing bandit algorithms and deep reinforcement learning algorithms that can be applied to MAB problems. An **epsilon-first** strategy begins with a pure exploration phase, selecting arms randomly for a fixed number of initial trials ($\epsilon$), before switching to a pure exploitation phase, repeatedly selecting the best-known arm. The **epsilon-greedy** algorithm balances exploration and exploitation by choosing a random arm with probability $\epsilon$ and the best-known arm with probability $1−\epsilon$ [4]. The **epsilon-decreasing** algorithm starts with a high exploration rate that decreases over time, allowing for more exploitation in later trials. **LinUCB** uses a linear model to estimate rewards based on contextual information and selects the arm with the highest upper confidence bound on the estimated reward [5]. **Thompson Sampling** is a Bayesian algorithm that models each arm's reward with a probability distribution, sampling from these distributions to decide which arm to pull, favoring arms with higher uncertainty [6]. In reinforcement learning, the **Actor-Critic** (A2C) method uses separate models for the policy (actor) and value function (critic), balancing immediate rewards and long-term value through stable learning. **Deep Q-Learning** uses deep neural networks to approximate Q-values (the expected reward of taking an action in a given state) and uses these approximations to guide action selection.


## Novel Algorithm: Dirichlet Forest Sampling

Existing algorithms for bounded reward distributions are either suboptimal or require solving an optimization problem at each step. Dirichlet parameters have been proposed as an extension of Thompson Sampling for Bernoulli rewards to bounded multinomial reward distributions, addressing the need for quick online decision-making without the computational burden of constant optimization [7]. Dirichlet parameters represent the probabilities associated with each possible reward level for each arm, allowing for efficient sampling and updating of beliefs. I extend this concept by integrating a random forest classifier to dynamically adjust Dirichlet parameters based on observed performance in contexts where we lack specific user data but know general tendencies (e.g., users having favorite genres). Specifically, in Netflix genre recommendations, it accelerates convergence to the optimal strategy, maximizing cumulative rewards more efficiently than traditional algorithms. Additionally, I propose a nonlinear update to the Dirichlet parameters, fitted to a reward of 1 for a value of 1 and 2 for a value of 5.

The algorithm proceeds as follows: initialize the Dirichlet parameters uniformly and set up the random forest classifier. At each time step, sample from the Dirichlet distribution to estimate the expected reward for each arm and select the arm with the highest expected reward. After pulling the selected arm and observing the reward, update the Dirichlet parameters with a nonlinear increment based on the observed reward. Periodically, train the random forest classifier on the accumulated reward history and boost the Dirichlet parameters for arms predicted to be favorites. Additionally, periodically adaptively boost the Dirichlet parameters based on observed average rewards exceeding a predefined threshold. 


\[
\begin{algorithm}
\caption{Dirichlet Forest Sampling}
\begin{algorithmic}
\REQUIRE \text{Number of trials} \( T \geq 1 \), \text{number of arms} \( K \geq 1 \), \text{reward levels} \( M = 5 \)
\STATE \text{Initialize Dirichlet parameters:} \( \alpha_k = [1, 1, 1, 1, 1] \) \text{for} \( k \in \{1, ..., K\} \)
\STATE \text{Initialize RandomForest model}
\FOR{\( t = 1 \) \TO \( T \)}
    \FOR{\( k = 1 \) \TO \( K \)}
        \STATE \text{Sample} \( L_k \sim \text{Dir}(\alpha_k) \)
        \STATE \text{Calculate expected reward} \( E_k = \sum_m \left( \frac{m}{M} \right) L_k[m] \)
    \ENDFOR
    \STATE \text{Select arm} \( I(t) = \arg\max_k E_k \)
    \STATE \text{Pull arm} \( I(t) \) \text{and observe reward} \( r_t \in \{1, ..., M\} \)
    \STATE \text{Update} \( \alpha_{I(t)}[r_t - 1] += 1 + 0.1 \times (r_t - 1) + 0.0375 \times (r_t - 1)^2 \)
    \IF{\( t \% N_{\text{forest}} == 0 \)}
        \STATE \text{Train RandomForest model on arm history}
        \STATE \text{Boost Dirichlet parameters using forest predictions}
    \ENDIF
    \IF{\( t \% N_{\text{adaptive}} == 0 \) \text{and} \( t \geq N_{\text{adaptive}} \)}
        \STATE \text{Adaptive boost: increase Dirichlet parameters for arms with average reward} \( > \theta \)
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
\]

IN PROGRESS...








1 - Explaining Recommendations: Leveraging User Feedback in a Bandit Framework
2 - Stochastic Bandits for Multi-Platform Budget Optimization in Online Advertising
3- Mao, Y., Chen, M., Wagle, A., Pan, J., Natkovich, M., Matheson, D. "A Batched Multi-Armed Bandit Approach to News Headline Testing." Oath Inc.
4 - Finite-time Analysis of the Multiarmed Bandit Problem.
5 - Lihong Li, Wei Chu, John Langford, Robert E. Schapire. "LinUCB: A Contextual-Bandit Approach to Personalized News Article Recommendation."
6 - W. R. Thompson, “On the likelihood that one unknown probability exceeds another in view of the evidence of two samples,” Biometrika, vol. 25, no. 3/4, pp. 285–294, 1933.
7 - Charles Riou, Junya Honda. "Bandit Algorithms Based on Thompson Sampling for Bounded Reward Distributions." Proceedings of the 31st International Conference on Algorithmic Learning Theory, PMLR 117:777-826, 2020.





