"""

Retrofitting algorithm steps:
1. Initialize the retrofitted word vectors as the pre-trained vectors.
2. Iterate over the word vectors, adjusting them based on their neighbors in the lexicon
3. Apply the update equation
4. Set the parameters alpha(i) and beta(ij) as per the baseline (same as step 1 initially)

Global steps:
1. Load the pre-trained GloVe model using gensim (the GloVe has been normalized alrdy, so we don't need to normalize it again)
2. Load the lexicon (our training set) from the provided files
3. Use the retrofitting function to adjust the vectors
4. Save the retrofitted word vectors

"""

import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import re
import copy

# Load GloVe pre-trained word vectors
model_gigaword = api.load("glove-wiki-gigaword-100")

# Read the lexicon (e.g., framenet, WordNet, PPDB or dev)
isNumber = re.compile(r'\d+.*')

def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()


def read_lexicon(filename):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[words[0]] = [word for word in words[1:]]
        # lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

# Retrofitting word vectors to a lexicon
def retrofit(model_gigaword, lexicon, alpha=1.0, beta=1.0, num_iters=10, verbose=True):

    Q_hat = {word: model_gigaword[word] for word in model_gigaword.index_to_key}

    Q = copy.deepcopy(Q_hat)
    Q_hat_keys = set(Q_hat.keys())

    found = set(lexicon.keys()).intersection(set(Q_hat.keys()))

    for ep in range(num_iters):
        print("epoch : ", ep)
        for idx, word in enumerate(found):
            print(idx, end='\r')
            word_synset = set(lexicon[word]).intersection(Q_hat_keys)
            if len(word_synset) == 0:
                continue

            qi = np.array([Q[ws] for ws in word_synset])
            sum_qi = np.sum(qi, axis=0)

            Q[word] = ((beta * sum_qi) + (alpha * Q_hat[word])) / ((beta * len(qi)) + alpha) # update

    return Q


# Save the retrofitted vectors
def save_word_vectors(word_vecs, filename):
    num_words = len(word_vecs.keys())
    vector_size = len(word_vecs['the'])

    with open(filename, 'w') as f:
        f.write(f"{num_words} {vector_size}\n")

        for word in word_vecs.keys():
            vector = word_vecs[word]
            vector_str = ' '.join(map(str, vector))
            f.write(f"{word} {vector_str}\n")


# Main function to run the retrofitting process
if __name__ == '__main__':
    lexicon = read_lexicon('../data/train/train/from_wikipedia.txt')
    retrofitted_vectors = retrofit(model_gigaword=model_gigaword, lexicon=lexicon, alpha=1.0, beta=1.0, num_iters=10, verbose=True)
    save_word_vectors(retrofitted_vectors, 'retrofitted_vectors_fromwiki.txt')