
#%% 
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

class DataLoader3:
    def __init__(self, file_path):
        self.file_path = file_path
        self.movies_df = None
        self.final_df = None
        self.genre_counts_df = None
        # self.action_df = None

        nltk.download('stopwords')

    def load_data(self):
        self.movies_df = pd.read_csv(self.file_path)
        self.movies_df['overview'] = self.movies_df.get('overview', '').fillna('')
        # Use a regular expression to split the genres on commas and any kind of space
        self.movies_df['genre'] = self.movies_df['genre'].apply(lambda x: re.split(r',\s*', x) if isinstance(x, str) else [])
        self.movies_df['sid'] = range(len(self.movies_df))

    def process_data(self):
        # Explode the genres column to create separate rows for each genre
        exploded_genres = self.movies_df.explode('genre')
        exploded_genres.dropna(subset=['genre'], inplace=True)

        # Filter to include only specified genres
        filtered_genres = exploded_genres[exploded_genres['genre'].isin(['Drama', 'Comedy', 'Action', 'Thriller', 'Adventure', 'Horror'])]

        # Group by sid and genre, then aggregate
        grouped_df = filtered_genres.groupby(['sid'], as_index=False).agg({'names': 'first', 'overview': 'first', 'genre': 'first'})

        

        # Now, limit to 1500 rows per genre
        final_df = grouped_df.groupby('genre').head(800).reset_index(drop=True)

        # Map genres to ids
        unique_genres = final_df['genre'].unique()
        print(unique_genres)
        genre_to_id = {genre: idx for idx, genre in enumerate(unique_genres)}
        final_df['genre_id'] = final_df['genre'].map(genre_to_id)

        # Apply any additional processing like removing stop words
        final_df['overview'] = final_df['overview'].apply(self.remove_stop_words)

        self.final_df = final_df

    @staticmethod
    def remove_stop_words(overview):
        overview_minus_sw = []
        stop_words = stopwords.words('english')
        overview = overview.lower().split()
        final_overview = [overview_minus_sw.append(word) for word in overview if word not in stop_words]
        final_overview = ' '.join(overview_minus_sw)
        return final_overview

    def analyze_genre_counts(self):
        genre_counts = self.final_df['genre'].value_counts()  # Changed line
        self.genre_counts_df = genre_counts.reset_index()
        self.genre_counts_df.columns = ['Genre', 'Count']

    def get_final_df(self):
        return self.final_df
    
    def get_genre_counts_df(self):
        return self.genre_counts_df
#%%
# 
# Usage)
'''
data_loader = DataLoader3('data/imdb_movies.csv')
data_loader.load_data()
data_loader.process_data()

final_df = data_loader.get_final_df()
data_loader.analyze_genre_counts()
genre_counts = data_loader.get_genre_counts_df()
# action_df = data_loader.get_action_df()
print(final_df)
# Now you can use final_df, genre_counts_df, and action_df as needed.
'''
# %%
