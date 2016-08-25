import os.path

from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

STOPWORDS = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

flatten = lambda ar: [a for b in ar for a in b]


def build_vocab(_posts, _num_words=None):
    """
    Build a vocabulary for all posts.  This can be used to or to limit the number of features,
    or produce more condensed data structures.
    """
    # TODO: rank words by TFIDF instead of frequency!
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
    # TODO: look into "internet savvy"/"Twitter aware" tokenizers - these can include smiley faces,
    #  slang etc.
    tokens = word_tokenize(_text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmas


def read_and_clean_posts(_fi_path, _num_words=None):
    """
    Read a file containing just the raw text of a post on each line, and return a nested list of
    the cleansed lemmas.

    If num_words is provided, the vocabulary is limited to the _num_words most frequent words not in
    the STOPWORDS list
    """
    posts = list(open(_fi_path, "r").readlines())
    posts = [s.strip() for s in posts]
    lemma_lists = [clean_raw_text(post) for post in posts]
    if _num_words:
        vocab = build_vocab(lemma_lists, _num_words)
        lemma_lists = [[l for l in ll if l in vocab] for ll in lemma_lists]
    return lemma_lists


def load_data_and_labels(_num_words=None, _positive_path="elefriends-1/details_of_sh-raw.txt", _negative_path="elefriends-1/unflagged-raw.txt"):
    """
    Run the whole data preparation pipeline

    :param num_words: maximum number of word features to include in the vocabulary.  Set if
    memory or performance is a problem
    """
    DATA_PATH = "/Volumes/data/Mind/data_dumps"
    POSITIVE = os.path.join(DATA_PATH, _positive_path)
    NEGATIVE = os.path.join(DATA_PATH, _negative_path)

    # Load data from files
    positive_examples = read_and_clean_posts(POSITIVE, _num_words)
    negative_examples = read_and_clean_posts(NEGATIVE, _num_words)

    positive_features = [post_features(p) for p in positive_examples]
    negative_features = [post_features(n) for n in negative_examples]

    _labels = ['p' for _ in positive_features] + ['n' for _ in negative_features]

    return positive_features + negative_features, _labels
