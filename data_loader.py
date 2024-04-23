import pandas as pd

# Read in subset of full Netflix prize data as DataFrame
def load_subset(data_path, verbose=False):
    train_path = data_path + "training_ratings.txt"
    test_path = data_path + "testing_ratings.txt"
    title_path = data_path + "movie_titles.txt"
    genre_path = data_path + "movie_genres.csv"

    # Read in movie titles line by line
    titles = []
    with open(title_path, 'r', encoding='latin1') as file:
        for line in file:
            # Maximum of 2 splits per line, avoiding commas in movie name
            parts = line.strip().split(',', 2)
            # Verify succesful split
            if len(parts) == 3:
                titles.append(parts)
            else:
                print(f"Skipped malformed line: {line}")

    # Create DataFrame for raw data
    df_titles = pd.DataFrame(titles, columns=['MovieID', 'Year', 'Title'])
    df_train = pd.read_csv(train_path, header=None, names=["MovieID", "UserID", "Rating"])
    df_test = pd.read_csv(test_path, header=None, names=["MovieID", "UserID", "Rating"])

    # Combine similar formatted data, and set correct data types for later merges
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_genres = pd.read_csv(genre_path, names=["MovieID", "Genres"]).drop(index=0)
    df_combined['MovieID'] = df_combined['MovieID'].astype(int)
    df_genres['MovieID'] = df_genres['MovieID'].astype(int)
    df_titles['MovieID'] = df_titles['MovieID'].astype(int)

    # Merge datasets
    df_merge1 = pd.merge(df_combined, df_genres, on='MovieID', how='inner')
    df_subset = pd.merge(df_merge1, df_titles, on='MovieID', how='inner')

    if verbose:
        # Check entries per user
        entries_per_user = df_subset.groupby('UserID').size()
        smallest_entries_count = entries_per_user.min()
        print(f"The smallest amount of entries for any UserID is: {smallest_entries_count}")

    return df_subset

# Convert large .txt file into structured DataFrame
def transform_file_to_dataframe(file_path):
    transformed_data = []
    with open(file_path, 'r') as file:
        current_movie_id = None  
        for line in file:
            line = line.strip()
            if line.endswith(':'):  
                # If current line is movie ID, strip colon to capture ID
                current_movie_id = line[:-1]  
            else:
                # Otherwise, split by commas and prepend most recent ID
                user_id, rating, date = line.split(',')
                transformed_data.append([current_movie_id, user_id, rating, date])
                
    columns = ['MovieID', 'UserID', 'Rating', 'Date']
    dataframe = pd.DataFrame(transformed_data, columns=columns)
    return dataframe

# Read full Netflix prize data as DataFrame 
def load_netflix_data(data_path, verbose=False):
    title_path = data_path + "movie_titles.txt"
    genre_path = data_path + "movie_genres.csv"
    
    # Read in movie titles line by line
    titles = []
    with open(title_path, 'r', encoding='latin1') as file:
        for line in file:
            # Maximum of 2 splits per line, avoiding commas in movie name
            parts = line.strip().split(',', 2)
            # Verify succesful split
            if len(parts) == 3:
                titles.append(parts)
            else:
                print(f"Skipped malformed line: {line}")
    
    df_titles = pd.DataFrame(titles, columns=['MovieID', 'Year', 'Title'])
    df_genres = pd.read_csv(genre_path, names=["MovieID", "Genres"]).drop(index=0)
    df_genres['MovieID'] = df_genres['MovieID'].astype(int)
    df_titles['MovieID'] = df_titles['MovieID'].astype(int)
    
    # Full datasets available at https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data
    df_1 = transform_file_to_dataframe(data_path + "combined_data_1.txt")
    df_2 = transform_file_to_dataframe(data_path + "combined_data_2.txt")
    df_3 = transform_file_to_dataframe(data_path + "combined_data_3.txt")
    df_4 = transform_file_to_dataframe(data_path + "combined_data_4.txt")
    
    # Merge datasets
    df_12 = pd.concat([df_1, df_2])
    df_34 = pd.concat([df_3, df_4])
    df_1234 = pd.concat([df_12, df_34])
    df_1234["MovieID"] = df_1234["MovieID"].astype(int)
    df_1234_genres = pd.merge(df_1234, df_genres, on="MovieID", how="left")
    df_titles["MovieID"] = df_titles["MovieID"].astype(int)
    df_final = pd.merge(df_1234_genres, df_titles, on="MovieID", how="left")

    if verbose:
        # Check entries per user
        entries_per_user = df_final.groupby('UserID').size()
        smallest_entries_count = entries_per_user.min()
        print(f"The smallest amount of entries for any UserID is: {smallest_entries_count}")

    return df_final

def preprocess(subset, verbose=False):
    # Split genres into lists and expand each genre into seprate row
    subset['Genres'] = subset['Genres'].str.split('|') 
    df_exploded = subset.explode('Genres') 
    genres = df_exploded["Genres"].unique().tolist()

    # Process genre-specific rating distributions before passing to environments
    unnormalized_distributions = {}
    for genre in genres:
        genre_ratings = list(df_exploded[df_exploded["Genres"] == genre]["Rating"])
        rating_counts = {}
        for rating in range(1,6):
            rating_counts[rating] = genre_ratings.count(rating)
        unnormalized_distributions[genre] = rating_counts

    # Determine niche genres based on threshold
    movies_per_genre = df_exploded.groupby('Genres')['MovieID'].nunique()
    threshold_movies = movies_per_genre.quantile(0.25)
    niche_genres = movies_per_genre[movies_per_genre <= threshold_movies].index.tolist()
    
    if verbose:
        print("Preprocess completed.")
        print("Genres:", genres)
        print("Niche Genres:", niche_genres)
        
    return genres, unnormalized_distributions, niche_genres