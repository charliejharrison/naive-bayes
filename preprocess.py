"""<DOCSTRING>
"""

__author__ = 'wah'

import os.path.join

from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

STOPWORDS = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


flatten = lambda ar: [a for b in ar for a in b]


def build_vocab(_posts, _num_words=None):
    """
    Build a vocabulary for all posts.  Not currently used...

    Need to check the total number of words in the corpus and consider using all vs subset.
    """
    vocab = FreqDist(flatten(_posts))
    vocab = list(vocab)[:_num_words]
    return vocab


def post_features(_post):
    """
    Generate a feature dictionary for a single post, for use with NLTK's naive Bayes classifier

    v0.1: Boolean bag of words - words map to True
    """
    features = {word: True for word in _post}
    return features


def clean_raw_text(_text):
    """
    From the raw text of a post: lower case, tokenize, remove stopwords,
    and lemmatize
    """
    # TODO: look into "internet savvy"/"Twitter aware" tokenizers - these can
    #  include smiley faces, slang etc.
    tokens = word_tokenize(_text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmas


def read_and_clean_posts(_fi_path):
    """
    Read a file containing just the raw text of a post on each line,
    and return a nested list of the cleansed lemmas
    """
    posts = list(open(_fi_path, "r").readlines())
    posts = [s.strip() for s in posts]
    lemma_lists = [clean_raw_text(post) for post in posts]
    return lemma_lists


def load_data_and_labels():
    """
    Run the whole data preparation pipeline
    """
    DATA_PATH = "/Volumes/data/Mind/data_dumps"
    POSITIVE = os.path.join(DATA_PATH, "details_of_sh-raw.txt")
    NEGATIVE = os.path.join(DATA_PATH, "unflagged-raw.tsv")

    # Load data from files
    positive_examples = read_and_clean_posts(POSITIVE)
    negative_examples = read_and_clean_posts(NEGATIVE)

    positive_features = [post_features(p) for p in positive_examples]
    negative_features = [post_features(n) for n in negative_examples]

    _labels = ['p' for _ in positive_features] + ['n' for _ in negative_features]

    return positive_features + negative_features, _labels
