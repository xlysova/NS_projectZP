#%%
from data_loader_v3 import DataLoader3
from sklearn.model_selection import train_test_split 
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data_loader = DataLoader3('data/imdb_movies.csv')
data_loader.load_data()
data_loader.process_data()
data_loader.analyze_genre_counts()

final_df = data_loader.get_final_df()
genre_counts_df = data_loader.get_genre_counts_df()
#action_df = action_df[~action_df['genre'].isin(['Animation', 'TV Movie', 'Family'])]
print(final_df)

#%%
# action_df.to_csv('final_data.csv', index=False)

# %%
X = final_df['overview']
y = array(final_df['genre_id'])

final_df.reset_index(inplace=True, drop=True)

unique_genres = final_df['genre_id'].unique()
y = to_categorical(y, num_classes=len(unique_genres))
X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)

# %%
oov_token = "<OOV>"
tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

# %%
word_index = tokenizer.word_index

# %%
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# %%
padding_type = "post"
trunction_type="post"

# Find the length of each sequence
sequence_lengths_train = [len(seq) for seq in X_train_sequences]
sequence_lengths_test = [len(seq) for seq in X_test_sequences]
sequence_lengths = sequence_lengths_train + sequence_lengths_test
# Determine the maximum length
max_length = max(sequence_lengths)
# max_length = 512

# %%
X_train_padded = pad_sequences(X_train_sequences,maxlen=max_length, padding=padding_type, truncating=trunction_type)

# %%
X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length, padding=padding_type,
                              truncating=trunction_type)

# %%
tf.random.set_seed(0)

# %%
training_data = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train))
validation_data = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test))

# %%
batch_size = 32
training_data = training_data.batch(batch_size)
validation_data = validation_data.batch(batch_size)

#%%
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# %%
epochs = 8

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers

vocab_size = len(tokenizer.word_index) + 1
# vocab_size = 5000
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.25)),
    Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.25)),
    Dense(len(unique_genres), activation='softmax', kernel_regularizer=regularizers.l2(0.01),
          activity_regularizer=regularizers.l1(0.01))  # number_of_genres should be set to the unique count of genres
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(training_data, epochs=epochs, verbose=1,validation_data = validation_data, callbacks = [callback])

# %%
loss, accuracy = model.evaluate(training_data)
print('Training Accuracy is {}'.format(accuracy*100))

#%%
loss, accuracy = model.evaluate(validation_data)
print('Testing Accuracy is {} '.format(accuracy*100))

# %%
import pandas as pd
hist = pd.DataFrame(history.history)

# %%
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,6))
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# hist['loss'].plot(label='Training Loss')
# hist['val_loss'].plot(label='Validation Loss')
# plt.legend(loc='center')

# %%
# plt.figure(figsize=(7,6))
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# hist['acc'].plot(label='Training Accuracy')
# hist['val_acc'].plot(label='Validation Accuracy')
# plt.legend(loc='center')

#%%
import matplotlib.pyplot as plt

# Assuming history is the return value from model.fit()
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if len(y_test.shape) > 1:  # This checks if y_test is one-hot encoded
    y_test = np.argmax(y_test, axis=1)
# Generate predictions for multi-class classification
y_pred_probs = model.predict(X_test_padded)
# Assuming y_test is a 1D array of true labels and y_pred_probs are the predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to predicted class labels

genre_mapping = final_df[['genre_id', 'genre']].drop_duplicates().sort_values('genre_id')
genre_labels = genre_mapping['genre'].values.tolist()
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_labels, yticklabels=genre_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# %%
import random

# Assuming 'title' is the column in final_df with the movie titles
# and 'genre_name' is the column in final_df with the genre names
# We need to make sure that X_test's index aligns with final_df

# Mapping from genre_id to genre_name
id_to_genre = genre_mapping.set_index('genre_id')['genre'].to_dict()

# Display some random predictions for multi-class
num_examples = 5
example_indices = random.sample(range(len(X_test)), num_examples)

BOLD_START = "\033[1m"
BOLD_END = "\033[0m"
TICK = u'\u2713'
CROSS = u'\u2717'

for idx in example_indices:
    movie_title = final_df.loc[X_test.index[idx], 'names']
    actual_genre_id = final_df.loc[X_test.index[idx], 'genre_id']
    actual_genre_name = id_to_genre[actual_genre_id]
    predicted_genre_id = y_pred[idx]
    predicted_genre_name = id_to_genre[predicted_genre_id]

    is_correct = actual_genre_id == predicted_genre_id
    correctness_text = (f"{BOLD_START}Correct {TICK}{BOLD_END}" if is_correct 
                        else f"{BOLD_START}Incorrect {CROSS}{BOLD_END}")

    # Print the information with formatted correctness text
    print(f"{BOLD_START}{movie_title}{BOLD_END}")
    print(f"Actual Genre: {actual_genre_name}")
    print(f"Predicted Genre: {predicted_genre_name}")
    print(f"Result: {correctness_text}")
    print("------------------------------------------")

# %%
