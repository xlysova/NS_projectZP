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

# Get unique genres and create a mapping from genre to genre_id
unique_genres = final_df['genre'].unique()
genre_to_id = {genre: idx + 1 for idx, genre in enumerate(unique_genres)}

# Add the 'genre_id' column using the mapping
final_df['genre_id'] = final_df['genre'].map(genre_to_id)

# Display the result
# print(final_df)
print(final_df[['sid', 'title', 'genre', 'genre_id']].head(20))

# %%
import nltk # Natural Language Toolkit 
nltk.download('stopwords') 
from nltk.corpus import stopwords

# Remove stop words
def remove_stop_words(overview):
    overview_minus_sw = []
    stop_words = stopwords.words('english')
    overview = overview.split()
    final_overview = [overview_minus_sw.append(word) for word in overview if word not in stop_words]            
    final_overview = ' '.join(overview_minus_sw)
    return final_overview   

#%%
final_df['overview'] = final_df['overview'].apply(remove_stop_words)

# %%
