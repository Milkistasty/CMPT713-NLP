import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
import math

#### Code from P. Norvig's book chapter in "Beautiful Data"

def memo(f):
    "Memoize function f."
    table = {}
    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo

#### Word Segmentation (p. 223)

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    @memo
    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        candidates = ([first]+self.segment(rem) for first,rem in self.splits(text))
        return max(candidates, key=self.Pwords)

    def splits(self, text, L=20):
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:]) 
                for i in range(min(len(text), L))]

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

def avoid_long_words(key, N):
    "Estimate the probability of an unknown word based on its length."
    "key represents a word or token from the input text that needs to be segmented or assigned a probability. It could be any string, including numbers, words, or unknown sequences"
    "N represents the total number of tokens in the training data or corpus (i.e., the sum of all word counts). It's used to calculate the probability of unknown words or sequences that are not found in the corpus"
    return 10./(N * 10**len(key))

# Penalizing based on the length of the unknown word
def return_scaled_prob_for_missing_word_with_int_handling(key,N):
    try:
        int(key)
        return 1e-10
    except:
        return (1./N)**len(key)

# zero if the words are not present
def return_zero_prob_for_missing_word_with_int_handling(key,N):
    try:
        int(key)
        return 1e-10
    except:
        return 0

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    sys.setrecursionlimit(10**6)

    # Pw = Pdist(data=datafile(opts.counts1w), missingfn=avoid_long_words)
    Pw = Pdist(data=datafile(opts.counts1w), missingfn=return_scaled_prob_for_missing_word_with_int_handling)
    segmenter = Segment(Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
