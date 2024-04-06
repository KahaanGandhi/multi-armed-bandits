import pandas as pd

def load_subset(data_path, check=False):
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

    if check:
        # Check entries per user
        entries_per_user = df_subset.groupby('UserID').size()
        smallest_entries_count = entries_per_user.min()
        print(f"The smallest amount of entries for any UserID is: {smallest_entries_count}")

    return df_subset