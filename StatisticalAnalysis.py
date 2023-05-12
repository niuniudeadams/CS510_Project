## Author: Jingyao Gu
## Graph adapted from https://www.kaggle.com/code/xxxxyyyy80008/analyze-co-occurrence-and-networks-of-words


import json
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter
from nltk.stem import WordNetLemmatizer
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rake_nltk import Rake

# Load the JSON file
with open('consented_submissions.json', 'r') as f:
    data = json.load(f)

text = ''
for obj in data:
    # text += obj['highlighted_text'] + ' ' + obj['explanation'] + ' '
    text += obj['highlighted_text']


def extract_keywords(text, n=10):
    # Tokenize the text and remove non-alphabetic characters and stop words
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    filtered_tokens = [token for token in filtered_tokens if len(token) > 1 and token not in stop_words]

    # Count the frequency of each word
    word_counts = {}
    for token in filtered_tokens:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1
            
    # Sort the words by frequency and select the top N
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word[0] for word in sorted_words[:n]]

# Extract keywords from the concatenated text
keywords = extract_keywords(text, n=50)
print("The most frequent keywords from all the submission are: ")
print(keywords)



def extract_keyword_pairs(text, n):
    # Tokenize the text and remove non-alphabetic characters and stop words
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    filtered_tokens = [token for token in filtered_tokens if len(token) > 1 and token not in stop_words]

    # Count the frequency of each word and each pair of adjacent words
    word_counts = Counter(filtered_tokens)
    pair_counts = Counter(zip(filtered_tokens, filtered_tokens[1:]))

    # Sort the pairs by frequency and select the top N
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    return [pair for pair in sorted_pairs[:n]]


# Extract pairs of adjacent keywords from the concatenated text
keyword_pairs = extract_keyword_pairs(text, n=30)
print("keywords pairs:")
print(keyword_pairs)


# Define a function to extract word triplets from a text string
def extract_word_triplets(text, n=10):
    # Tokenize the text and remove non-alphabetic characters and stop words
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    filtered_tokens = [token for token in filtered_tokens if len(token) > 1 and token not in stop_words]

    # Count the frequency of each word and each triplet of adjacent words
    word_counts = Counter(filtered_tokens)
    triplet_counts = Counter(zip(filtered_tokens, filtered_tokens[1:], filtered_tokens[2:]))

    # Sort the triplets by frequency and select the top N
    sorted_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)
    return [triplet for triplet in sorted_triplets[:n]]

# Extract top 10 word triplets from the concatenated text
word_triplets = extract_word_triplets(text, n=20)
print("triplets")
print(word_triplets)

# Create network plot 
G = nx.Graph()

# Create connections between nodes
for v in keyword_pairs:
    G.add_edge(v[0][0], v[0][1], weight=(v[1] * 10))


fig, ax = plt.subplots(figsize=(18, 10))

pos = nx.spring_layout(G, k=8)

d = dict(nx.degree(G))
edges = G.edges()
weights = [G[u][v]['weight']/500 for u,v in edges]
# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=weights,
                 node_size = [v * 200 for v in d.values()], 
                 edge_color='grey',
                 #node_color='tomato',
                 with_labels = True,
                 ax=ax)

ax.set_title('Bigram Network', 
             fontdict={'fontsize': 16,
            'fontweight': 'bold',
            'color': 'salmon', 
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, 
             loc='center')    
# plt.show()







from nltk.tokenize import sent_tokenize, word_tokenize
# Load the JSON file
with open('consented_submissions.json', 'r') as f:
    data = json.load(f)

# Concatenate all the "highlighted_text" and "explanation" fields into a single string
text = ''
for obj in data:
    text += obj['highlighted_text'] + ' ' + obj['explanation'] + ' '

text = text.replace('lecture', '')
# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Define a function to extract word pairs from a sentence
def extract_word_pairs(sentence):
    # Tokenize the sentence and remove non-alphabetic characters and stop words
    tokens = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    filtered_tokens = [token for token in filtered_tokens if len(token) > 1 and token not in stop_words]

    # Generate pairs of words that co-occur in the same sentence
    word_pairs = combinations(filtered_tokens, 2)
    return list(word_pairs)

# Extract word pairs from all the sentences
pairs = []
for sentence in sentences:
    pairs += extract_word_pairs(sentence)

# Count the frequency of each word pair
pair_counts = Counter(pairs)

# Sort the pairs by frequency and select the top 50
sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
top_pairs = [(pair[0], pair[1]) for pair in sorted_pairs[:100]]


# Create network plot 
G = nx.Graph()

# Create connections between nodes
for v in top_pairs:
    G.add_edge(v[0][0], v[0][1], weight=(v[1] * 10))


fig, ax = plt.subplots(figsize=(15, 10))

pos = nx.spring_layout(G, k=8)

d = dict(nx.degree(G))
edges = G.edges()
weights = [G[u][v]['weight']/2000 for u,v in edges]
# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=weights,
                 node_size = [v * 200 for v in d.values()], 
                 edge_color='grey',
                 #node_color='tomato',
                 with_labels = True,
                 ax=ax)

ax.set_title('Co-occurance Network', 
             fontdict={'fontsize': 16,
            'fontweight': 'bold',
            'color': 'salmon', 
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center'}, 
             loc='center')    
plt.show()

