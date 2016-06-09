"""<DOCSTRING>
"""

import os.path.join

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk import NaiveBayesClassifier

from collections import Counter


STOPWORDS = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

__author__ = 'wah'

# Read and clean
# - Stopwords
# - Stem
# - Bag of words feature matrix

# - Start with all-vs-all
# - Then try multiclass n-hot


flatten = lambda ar: [a for b in ar for a in b]


def build_vocab(posts, num_words=None):
    """
    Build a vocabulary for all posts.  Not currently used...

     Need to check the total number of words in the corpus and consider using all vs subset.

    :return:
    """
    vocab = FreqDist(flatten(posts))
    vocab = list(vocab)[:num_words]
    return vocab


def post_features(post):
    """
    Boolean bag of words - words map to True

    :param post:
    :return:
    """
    features = {"has({})".format(word) for word in post}
    return features


def clean_raw_text(text):
    """
    From the raw text of a post: lower case, tokenize, remove stopwords,
    and lemmatize

    :param post:
    :return:
    """
    # TODO: look into "internet savvy"/"Twitter aware" tokenizers - these can
    #  include smiley faces, slang etc.
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmas


def read_and_clean_posts(fi_path):
    """
    Read a file containing just the raw text of a post on each line,
    and return a nested list of the cleansed lemmas

    :param post:
    :return:
    """
    posts = list(open(fi_path, "r").readlines())
    posts = [s.strip() for s in posts]
    lemma_lists = [clean_raw_text(post) for post in posts]
    return lemma_lists


def load_data_and_labels():
    """
    Wrapper method for running cleansing pipeline
    :return:
    """
    DATA_PATH = "/Volumes/data/Mind/data_dumps"
    POSITIVE = os.path.join(DATA_PATH, "details_of_sh-raw.txt")
    NEGATIVE = os.path.join(DATA_PATH, "unflagged-raw.tsv")

    # Load data from files
    positive_examples = read_and_clean_posts(POSITIVE)
    negative_examples = read_and_clean_posts(NEGATIVE)

    positive_features = [post_features(p) for p in positive_examples]
    negative_features = [post_features(n) for n in negative_examples]

    documents = [(p, 'POSITIVE') for p in positive_features] + \
                [(n, 'NEGATIVE') for n in negative_features]

    return documents


if __name__ == '__main__':
    documents = load_data_and_labels()

    # Split into train and test set
    classifier = NaiveBayesClassifier().train(train_set)