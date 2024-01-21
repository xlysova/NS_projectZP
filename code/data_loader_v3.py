
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
        self.all_genre_counts_df = None
        self.horror_df = None

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
        
        self.all_genre_counts_df = self.analyze_all_genre_counts(exploded_genres)
        # Filter to include only specified genres
        filtered_genres = exploded_genres[exploded_genres['genre'].isin(['Drama', 'Comedy', 'Action', 'Adventure', 'Horror'])]
        # Group by sid and genre, then aggregate
        # grouped_df = filtered_genres.groupby(['sid'], as_index=False).agg({'names': 'first', 'overview': 'first', 'genre': 'first'})
        # df for data analysis
        grouped_df = filtered_genres.groupby(['sid'], as_index=False).agg({'names': 'first', 'genre': 'first', 'overview': 'first', 'country': 'first'})

        # Now, limit to 1500 rows per genre
        final_df = grouped_df.groupby('genre').head(1000).reset_index(drop=True)

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
        other_stop_words = ['new', 'find', 'one', 'must', 's ', 'young', 'two']
        stop_words.extend(other_stop_words)
        # print(stop_words)
        overview = overview.lower().split()
        final_overview = [overview_minus_sw.append(word) for word in overview if word not in stop_words and not word.isdigit()]
        final_overview = ' '.join(overview_minus_sw)
        return final_overview

    def analyze_genre_counts(self):
        genre_counts = self.final_df['genre'].value_counts()  # Changed line
        self.genre_counts_df = genre_counts.reset_index()
        self.genre_counts_df.columns = ['Genre', 'Count']

    def analyze_all_genre_counts(self, df):
        genre_counts = df['genre'].value_counts()  # Changed line
        self.all_genre_counts_df = genre_counts.reset_index()
        self.all_genre_counts_df.columns = ['Genre', 'Count']
        return self.all_genre_counts_df

    def analyze_chosen_genre(self):
        horror_df = self.final_df.copy()
        horror_df['is_horror'] = horror_df['genre'] == 'Horror'  # Changed line
        horror_df = horror_df.groupby('sid', as_index=False).agg({'overview': 'first', 'sid': 'first', 'names': 'first', 'is_horror': 'any'})
        horror_df['is_horror'] = horror_df['is_horror'].astype(int)
        self.horror_df = horror_df
        self.horror_df['overview'] = self.horror_df['overview'].apply(self.remove_stop_words)

    def get_final_df(self):
        return self.final_df
    
    def get_genre_counts_df(self):
        return self.genre_counts_df
    
    def get_original_df(self):
        return self.movies_df[['sid', 'names', 'genre', 'overview']]
    
    def get_all_genre_counts_df(self):
        return self.all_genre_counts_df
    
    def get_horror_df(self):
        return self.horror_df
#%%

# Usage)

data_loader = DataLoader3('data/imdb_movies.csv')
data_loader.load_data()
data_loader.process_data()

final_df = data_loader.get_final_df()
original_df = data_loader.get_original_df()

genre_counts = data_loader.get_genre_counts_df()
all_genre_counts = data_loader.get_all_genre_counts_df()
data_loader.analyze_chosen_genre()
horror_df = data_loader.get_horror_df()
print(print(horror_df[horror_df['is_horror'] == 1]))
# Now you can use final_df, genre_counts_df, and action_df as needed.

# %%
