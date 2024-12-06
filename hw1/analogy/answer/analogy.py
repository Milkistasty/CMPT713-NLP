import gensim.downloader as api
from gensim.models import KeyedVectors
import sys

# model_gigaword = api.load("glove-wiki-gigaword-100")

# Load the retrofitted vectors

retrofitted_model = KeyedVectors.load_word2vec_format('answer/retrofitted_vectors_fromwiki.txt', binary=False)

for i, line in enumerate(sys.stdin):
    line = line.strip()
    if line[0] == ':':
        print(line)
        continue
    (a, b, c) = line.split()

    # results = model_gigaword.most_similar(positive=[a.lower(), c.lower()], negative=[b.lower()])
    results = retrofitted_model.most_similar(positive=[a.lower(), c.lower()], negative=[b.lower()])

    print(results[0][0])
