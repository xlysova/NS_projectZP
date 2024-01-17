#%%
import pandas as pd

movies_df = pd.read_csv('data/movies.csv')
# Convert the 'genres' column to a list of dictionaries if it's a string representation
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['genres'] = movies_df['genres'].apply(ast.literal_eval)
movies_df['sid'] = range(len(movies_df))

#%%
import pandas as pd
import ast  # To safely evaluate strings containing Python expressions

# Assuming movies_df is your original DataFrame
# Example: movies_df = pd.read_csv('your_dataset.csv')

# Explode the genres
exploded_genres = movies_df.explode('genres')

# Extract genre names, handling NaN values
exploded_genres['genre'] = exploded_genres['genres'].apply(lambda x: x['name'] if pd.notna(x) else None)

# Drop rows where genre is None (if you want to exclude movies without a genre)
exploded_genres.dropna(subset=['genre'], inplace=True)

# Select and rename columns
final_df = exploded_genres[['sid', 'title', 'genre', 'overview']]

# Drop duplicates if necessary
final_df.drop_duplicates(inplace=True)

# Display the result
print(final_df)