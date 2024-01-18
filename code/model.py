#%%
from data_loader import DataLoader
from sklearn.model_selection import train_test_split 
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

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
X = action_df['overview']
y = array(action_df['is_action'])

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)
# %%
oov_token = "<OOV>"
tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(X_train)