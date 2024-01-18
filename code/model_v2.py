#%%
from data_loader import DataLoader
from sklearn.model_selection import train_test_split 
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.30)

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
epochs = 3

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM, Bidirectional
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

vocab_size = len(tokenizer.word_index) + 1
# vocab_size = 5000
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64,)),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
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
import matplotlib.pyplot as plt
plt.figure(figsize=(7,6))
plt.xlabel('Epochs')
plt.ylabel('Loss')
hist['loss'].plot(label='Training Loss')
hist['val_loss'].plot(label='Validation Loss')
plt.legend(loc='center')

# %%
plt.figure(figsize=(7,6))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
hist['acc'].plot(label='Training Accuracy')
hist['val_acc'].plot(label='Validation Accuracy')
plt.legend(loc='center')
# %%
