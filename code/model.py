#%%
from data_loader import DataLoader
from sklearn.model_selection import train_test_split 
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

data_loader = DataLoader('data/movies.csv')
data_loader.load_data()
data_loader.process_data()
data_loader.analyze_genre_counts()
data_loader.analyze_action_genre()

final_df = data_loader.get_final_df()
genre_counts_df = data_loader.get_genre_counts_df()
action_df = data_loader.get_action_df()
print(action_df)

# %%
X = action_df['overview']
y = array(action_df['is_action'])

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)
# %%
oov_token = "<OOV>"
tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
# %%
word_index = tokenizer.word_index

# %%
