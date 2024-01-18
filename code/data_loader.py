#%%
import pandas as pd
import ast
import nltk
from nltk.corpus import stopwords

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.movies_df = None
        self.final_df = None
        self.genre_counts_df = None
        self.action_df = None

        nltk.download('stopwords')

    def load_data(self):
        self.movies_df = pd.read_csv(self.file_path)
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['genres'] = self.movies_df['genres'].apply(ast.literal_eval)
        self.movies_df['sid'] = range(len(self.movies_df))

    def process_data(self):
        exploded_genres = self.movies_df.explode('genres')
        exploded_genres['genre'] = exploded_genres['genres'].apply(lambda x: x['name'] if pd.notna(x) else None)
        exploded_genres.dropna(subset=['genre'], inplace=True)
        self.final_df = exploded_genres[['sid', 'title', 'genre', 'overview']]
        self.final_df.drop_duplicates(inplace=True)

        unique_genres = self.final_df['genre'].unique()
        genre_to_id = {genre: idx + 1 for idx, genre in enumerate(unique_genres)}
        self.final_df['genre_id'] = self.final_df['genre'].map(genre_to_id)

        self.final_df['overview'] = self.final_df['overview'].apply(self.remove_stop_words)

    @staticmethod
    def remove_stop_words(overview):
        overview_minus_sw = []
        stop_words = stopwords.words('english')
        overview = overview.split()
        final_overview = [overview_minus_sw.append(word) for word in overview if word not in stop_words]            
        final_overview = ' '.join(overview_minus_sw)
        return final_overview

    def analyze_genre_counts(self):
        genre_counts = self.final_df['genre'].value_counts()
        self.genre_counts_df = genre_counts.reset_index()
        self.genre_counts_df.columns = ['Genre', 'Count']

    def analyze_action_genre(self):
        action_df = self.final_df.copy()
        action_df['is_action'] = action_df['genre_id'] == 1
        action_df = action_df.groupby('sid', as_index=False).agg({'overview': 'first', 'sid': 'first', 'is_action': 'any'})
        action_df['is_action'] = action_df['is_action'].astype(int)
        self.action_df = action_df



    def get_final_df(self):
        return self.final_df

    def get_genre_counts_df(self):
        return self.genre_counts_df

    def get_action_df(self):
        return self.action_df

#%%
# Usage
data_loader = DataLoader('data/movies.csv')
data_loader.load_data()
data_loader.process_data()
data_loader.analyze_genre_counts()
data_loader.analyze_action_genre()

final_df = data_loader.get_final_df()
genre_counts_df = data_loader.get_genre_counts_df()
action_df = data_loader.get_action_df()
print(action_df)
# Now you can use final_df, genre_counts_df, and action_df as needed.
# %%
