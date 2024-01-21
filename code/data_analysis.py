#%%
from data_loader_v3 import DataLoader3
data_loader = DataLoader3('data/imdb_movies.csv')

data_loader.load_data()
original_df = data_loader.get_original_df()

data_loader.process_data()
final_df = data_loader.get_final_df()
# words_to_delete = ['new', 'find', 'one', 'must', 's ', 'young', 'two']
# for word in words_to_delete:
#     final_df['overview'] = final_df['overview'].str.replace(word, '')

data_loader.analyze_genre_counts()
genre_counts_df = data_loader.get_genre_counts_df()
all_genre_counts_df = data_loader.get_all_genre_counts_df()

#%%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# List of unique genres
unique_genres = final_df['genre'].unique()

# Create a word cloud for each genre
for genre in unique_genres:
    # Filter data for the current genre
    genre_data = final_df[final_df['genre'] == genre]
    
    # Combine overviews into a single string
    overview_text = ' '.join(genre_data['overview'])
    
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(overview_text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {genre} Genre')
    plt.axis('off')
    plt.show()

# %%
from collections import Counter

# Load your data from final_df (replace 'final_df' with your actual DataFrame name)
# For example:
# final_df = pd.read_csv('your_data.csv')

# List of unique genres
unique_genres = final_df['genre'].unique()

# Combine overviews for all genres into a single string
all_genres_overviews = ' '.join(final_df['overview'])

# Count word occurrences for all genres
word_counts_all_genres = Counter(all_genres_overviews.split())

# Get the top 10 most common words across all genres
top_words_all_genres = word_counts_all_genres.most_common(5)

# Print the top common words across all genres
print("Slová, které jsou hodně používané napříč žánry:")
for word, count in top_words_all_genres:
    print(f'{word}: {count}')

# %%
final_df['overview'].describe()

# %%
def compute_len(overview):
    length = 0
    length = len(overview)
    return length

final_df['overview_len'] = final_df['overview'].map(compute_len)
# %%
final_df['overview_len'].describe()

# %%

import numpy as np
import matplotlib.pyplot as plt 

ax = all_genre_counts_df.plot(kind='bar', x='Genre', y='Count', figsize=(16, 10), color='skyblue', legend=False)
 
# Add some labels and a title
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Count of All Movies by Genre')

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height() + 10), ha='center', va='bottom')

plt.tight_layout()
# Show the plot
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt 

ax = genre_counts_df.plot(kind='bar', x='Genre', y='Count', figsize=(16, 10), color='skyblue', legend=False)
 
# Add some labels and a title
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Count of Chosen Movies by Genre')

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height() + 10), ha='center', va='bottom')

plt.tight_layout()
# Show the plot
plt.show()

# %%
