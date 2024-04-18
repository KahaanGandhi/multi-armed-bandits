import matplotlib.pyplot as plt
import numpy as np
    
#--------------------------------------------------------#
# Genre enjoyer: biases ratings towards a favorite genre
#--------------------------------------------------------#

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
        ratings, probabilities = zip(*self.reward_distributions[genre].items()) 
        reward = np.random.choice(ratings, p=probabilities)
        return reward
        
    # Plot the normalized rating distribution for each genre
    def plot_distributions(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.genre_list)))
        # Iterate through colors, with emphasis on the favorite 
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

#----------------------------------------------------#
# Niche genre loyalist: preference for a niche genre
#----------------------------------------------------#

class NicheGenreLoyalistEnvironment:
    def __init__(self, genre_list, niche_genre_list, unnormalized_distributions, user_id):
        self.genre_list = genre_list
        self.user_id = user_id
        self.favorite_genre = np.random.choice(niche_genre_list)  # Randomly select a favorite genre
        self.reward_distributions = self.create_reward_distributions(unnormalized_distributions)

    # Create stochastic and stationary reward distributions for each genre, biased towards favorite
    def create_reward_distributions(self, unnormalized_distributions):
        normalized_distributions = {}
        for genre in self.genre_list:
            rating_counts = unnormalized_distributions[genre].copy()
            if genre == self.favorite_genre:
                bias_factor = 1.5  
                rating_counts[4] = int(rating_counts[4] * bias_factor)
                rating_counts[5] = int(rating_counts[5] * bias_factor)
            else:
                nerf_factor = 0.7
                rating_counts[4] = int(rating_counts[4] * nerf_factor)
                rating_counts[5] = int(rating_counts[5] * nerf_factor)
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
        ratings, probabilities = zip(*self.reward_distributions[genre].items())
        reward = np.random.choice(ratings, p=probabilities)
        return reward
        
    # Plot the normalized rating distribution for each genre
    def plot_distributions(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.genre_list)))
        # Iterate through colors, with emphasis on the favorite 
        for genre, color in zip(self.genre_list, colors):
            if genre in self.reward_distributions:
                ratings = list(self.reward_distributions[genre].keys())
                probabilities = list(self.reward_distributions[genre].values())
                if genre == self.favorite_genre:
                    plt.plot(ratings, probabilities, '-o', label=genre + " (Fav)", color="red")
                else:
                    plt.plot(ratings, probabilities, '-o', label=genre, color=color)
        plt.title('Normalized Rating Distribution for Niche Genre Loyalist')
        plt.xlabel('Rating')
        plt.ylabel('Proportion')
        plt.xticks(range(1, 6))
        plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()

#----------------------------------------------------------------#
# Multiple genre enjoyer: biases ratings towards favorite genres
#----------------------------------------------------------------#

class MultipleGenreEnjoyerEnvironment:
    def __init__(self, genre_list, unnormalized_distributions, user_id):
        self.genre_list = genre_list
        self.user_id = user_id
        self.favorite_genres = np.random.choice(genre_list, size=np.random.randint(1, len(genre_list)), replace=False)
        self.reward_distributions = self.create_reward_distributions(unnormalized_distributions)

    # Create stochastic and stationary reward distributions for each genre, biased towards favorites
    def create_reward_distributions(self, unnormalized_distributions):
        normalized_distributions = {}
        for genre in self.genre_list:
            rating_counts = unnormalized_distributions[genre].copy()
            if genre in self.favorite_genres:
                # Apply a bias to ratings 4 and 5 for favorite genres
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
        ratings, probabilities = zip(*self.reward_distributions[genre].items())
        reward = np.random.choice(ratings, p=probabilities)
        return reward

    # Plot the normalized rating distribution for each genre with emphasis on favorites
    def plot_distributions(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.genre_list)))
        for genre, color in zip(self.genre_list, colors):
            if genre in self.reward_distributions:
                ratings = list(self.reward_distributions[genre].keys())
                probabilities = list(self.reward_distributions[genre].values())
                if genre in self.favorite_genres:
                    plt.plot(ratings, probabilities, '-o', label=genre + " (Fav)", color="red")
                else:
                    plt.plot(ratings, probabilities, '-o', label=genre, color=color)
        plt.title('Normalized Rating Distribution for Multiple Genre Enjoyer')
        plt.xlabel('Rating')
        plt.ylabel('Proportion')
        plt.xticks(range(1, 6))
        plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()

#------------------------------------------------------------------------#
# Multiple niche genre loyalist: preference for one or more niche genres
#------------------------------------------------------------------------#

class MultipleNicheGenreLoyalistEnvironment:
    def __init__(self, genre_list, niche_genre_list, unnormalized_distributions, user_id):
        self.genre_list = genre_list
        self.user_id = user_id
        self.num_boosted_genres = np.random.randint(1, len(niche_genre_list) + 1)
        self.boosted_genres = np.random.choice(niche_genre_list, size=self.num_boosted_genres, replace=False)
        self.reward_distributions = self.create_reward_distributions(unnormalized_distributions)

    # Create stochastic and stationary reward distributions for each genre, biased towards favorite
    def create_reward_distributions(self, unnormalized_distributions):
        normalized_distributions = {}
        for genre in self.genre_list:
            rating_counts = unnormalized_distributions[genre].copy()
            if genre in self.boosted_genres:
                bias_factor = 1.5  
                rating_counts[4] = int(rating_counts[4] * bias_factor)
                rating_counts[5] = int(rating_counts[5] * bias_factor)
            else:
                nerf_factor = 0.6
                rating_counts[4] = int(rating_counts[4] * nerf_factor)
                rating_counts[5] = int(rating_counts[5] * nerf_factor)
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
        ratings, probabilities = zip(*self.reward_distributions[genre].items())
        reward = np.random.choice(ratings, p=probabilities)
        return reward
        
    # Plot the normalized rating distribution for each genre
    def plot_distributions(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.genre_list)))
        # Iterate through colors, with emphasis on the favorite 
        for genre, color in zip(self.genre_list, colors):
            if genre in self.reward_distributions:
                ratings = list(self.reward_distributions[genre].keys())
                probabilities = list(self.reward_distributions[genre].values())
                if genre in self.boosted_genres:
                    plt.plot(ratings, probabilities, '-o', label=genre + " (Fav)", color="red")
                else:
                    plt.plot(ratings, probabilities, '-o', label=genre, color=color)
        plt.title('Normalized Rating Distribution for Multiple Niche Genre Loyalist')
        plt.xlabel('Rating')
        plt.ylabel('Proportion')
        plt.xticks(range(1, 6))
        plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()

#-----------------------------------------------------------------------#
# Average viewers: average ratings conform with genre-specific averages 
#-----------------------------------------------------------------------#

class AverageViewerEnvironment:
    def __init__(self, genre_list, unnormalized_distributions, user_id):
        self.genre_list = genre_list
        self.user_id = user_id
        self.reward_distributions = self.create_reward_distributions(unnormalized_distributions)

    # Create stochastic and stationary reward distributions for each genre
    def create_reward_distributions(self, unnormalized_distributions):
        normalized_distributions = {}
        for genre in self.genre_list:
            rating_counts = unnormalized_distributions[genre].copy()
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
        ratings, probabilities = zip(*self.reward_distributions[genre].items())
        reward = np.random.choice(ratings, p=probabilities)
        return reward
        
    # Plot the normalized rating distribution for each genre
    def plot_distributions(self):
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.genre_list)))
        # Iterate through colors
        for genre, color in zip(self.genre_list, colors):
            if genre in self.reward_distributions:
                ratings = list(self.reward_distributions[genre].keys())
                probabilities = list(self.reward_distributions[genre].values())
                plt.plot(ratings, probabilities, '-o', label=genre, color=color)        
        plt.title('Normalized Rating Distribution for Average Viewer')
        plt.xlabel('Rating')
        plt.ylabel('Proportion')
        plt.xticks(range(1, 6))
        plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()