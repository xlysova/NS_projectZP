
#%% 
import pandas as pd
import nltk
from nltk.corpus import stopwords

class DataLoader4:
    def __init__(self, file_path):
        self.file_path = file_path
        self.movies_df = None
        # self.final_df = None

        nltk.download('stopwords')

    def load_data(self):
        self.movies_df = pd.read_csv(self.file_path)
        self.movies_df['review'] = self.movies_df.get('review', '').fillna('')
        # self.movies_df['attitude'] = self.movies_df.get('attitude', '').fillna('')
        self.movies_df['sid'] = range(len(self.movies_df))

    def process_data(self):
        # Convert 'positive' to 1 and 'negative' to 0
        self.movies_df['attitude_id'] = self.movies_df['attitude'].map({'positive': 1, 'negative': 0})
        self.movies_df['review'] = self.movies_df['review'].apply(self.remove_stop_words)

    @staticmethod
    def remove_stop_words(overview):
        overview_minus_sw = []
        stop_words = stopwords.words('english')
        overview = overview.lower().split()
        final_overview = [overview_minus_sw.append(word) for word in overview if word not in stop_words]
        final_overview = ' '.join(overview_minus_sw)
        return final_overview

    def get_final_df(self):
        return self.movies_df

#%%
'''
# Usage
data_loader = DataLoader4('data/reviews.csv')
data_loader.load_data()
data_loader.process_data()

final_df = data_loader.get_final_df()
print(final_df)
# Now you can use final_df, genre_counts_df, and action_df as needed.
'''
# %%
