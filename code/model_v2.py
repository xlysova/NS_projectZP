#%%
from data_loader_v3 import DataLoader3
from sklearn.model_selection import train_test_split 
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_loader = DataLoader3('data/imdb_movies.csv')
data_loader.load_data()
data_loader.process_data()

final_df = data_loader.get_final_df()
original_df = data_loader.get_original_df()

genre_counts = data_loader.get_genre_counts_df()
all_genre_counts = data_loader.get_all_genre_counts_df()
data_loader.analyze_chosen_genre()
horror_df = data_loader.get_horror_df()
# print(print(horror_df[horror_df['is_horror'] == 1]))

# %%
X = horror_df['overview']
y = array(horror_df['is_horror'])

final_df.reset_index(inplace=True, drop=True)

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.30)

#%%
# from sklearn.utils import class_weight
# import numpy as np

# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights_dict = dict(enumerate(class_weights))

import pandas as pd

# Assuming 'df' is your DataFrame and it contains a column 'class_label' with values 0 and 1
# Replace 'class_label' with the actual name of your column

# Separate the minority and majority classes
df_majority = horror_df[horror_df['is_horror'] == 0]
df_minority = horror_df[horror_df['is_horror'] == 1]

# Upsample minority class
df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=123)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
horror_df = df_upsampled

# Display new class counts
print(horror_df['is_horror'].value_counts())
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
batch_size = 64
training_data = training_data.batch(batch_size)
validation_data = validation_data.batch(batch_size)
epochs = 16
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
vocab_size = len(tokenizer.word_index) + 1

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Bidirectional(LSTM(32, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Using a smaller learning rate
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

history = model.fit(training_data, epochs=epochs, verbose=1, validation_data=validation_data, callbacks=[callback])

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

# %%
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

# Generate predictions
y_pred_probs = model.predict(X_test_padded)
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary predictions

# Flatten y_pred to match the shape of y_test
y_pred = y_pred.flatten()

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
import random

# Display some random predictions
num_examples = 10
example_indices = random.sample(range(len(X_test)), num_examples)

BOLD_START = "\033[1m"
BOLD_END = "\033[0m"
TICK = u'\u2713'
CROSS = u'\u2717'

for idx in example_indices:
    movie_title = final_df.loc[X_test.index[idx], 'names']
    actual_genre_id = final_df.loc[X_test.index[idx], 'genre_id']
    predicted_genre_id = y_pred[idx]

    # Assuming that genre_id 1 is for Horror and 0 for Not Horror
    actual_genre_name = 'Horror' if actual_genre_id == 1 else 'Not Horror'
    predicted_genre_name = 'Horror' if predicted_genre_id == 1 else 'Not Horror'

    is_correct = actual_genre_id == predicted_genre_id
    correctness_text = (f"{BOLD_START}Correct {TICK}{BOLD_END}" if is_correct 
                        else f"{BOLD_START}Incorrect {CROSS}{BOLD_END}")

    print(f"{BOLD_START}Title: {movie_title}{BOLD_END}")
    print(f"Actual Genre: {actual_genre_name}")
    print(f"Predicted Genre: {predicted_genre_name}")
    print(f"Result: {correctness_text}")
    print("------------------------------------------")
# %%
