#%%
from data_loader import DataLoader
import pandas as pd

dl = DataLoader('data/movies.csv')
dl.load_data()
dl.process_data()
dl.analyze_genre_counts()
dl.analyze_action_genre()

final_df = dl.get_final_df()
genre_counts_df = dl.get_genre_counts_df()
action_df = dl.get_action_df()
print(action_df)

# %%
