from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib import pyplot as plt
import re
import matplotlib.cm as cm

NON_ALPHABETIC_REGEX = re.compile('[^a-zA-Z]')
STOP_WORDS = set(stopwords.words('english'))

class DocExplorer():
    """ Visualisation tool for static and dynamic exploration of documents. """

    def __init__(self, clustering=DBSCAN(), n_keywords_static=3, n_keywords_dynamic=5):
        """
        :param clustering: Clustering object with sklearn's interface (fit_predict method)
        :param method: Method to use for keyword extraction. Either 'tfidf' or 'g2'
        :param n_keywords_static: Number of keywords to display per cluster
        :param n_keywords_dynamic: Number of keywords to display for lense exploration
        """
        self.clustering = clustering
        self.n_keywords_static = n_keywords_static
        self.n_keywords_dynamic = n_keywords_dynamic

        self.tfidf_matrix = None
        self.tfidf_feature_names = None

        self.X_em = None

    def fit(self, docs, X_em=None):
        """
        Performs text preprocessing, feature extraction and embedding. Remembers what is needed for lens exploration.
        :param docs: list of text documents (strings)
        :param X_em: numpy array of embeddings with shape (n_samples, 2). If None, t-sne is performed on word count matrix.
        """

        def remove_non_alphabetic(text):
            return NON_ALPHABETIC_REGEX.sub(' ', text).lower()

        tfidf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                           preprocessor=remove_non_alphabetic,
                                           max_df=0.5)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(docs).toarray()
        self.tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())

        if X_em is None:
            self.X_em = TSNE().fit_transform(self.tfidf_matrix)
        else:
            self.X_em = X_em

    def plot_static(self):
        """
        Plots static labels for clusters of the embedding.
        """
        clusters = self.clustering.fit_predict(self.X_em)
        cmap = cm.get_cmap('Set1')
        for c in np.unique(clusters):
            is_in_cluster = clusters==c
            plt.scatter(self.X_em[is_in_cluster, 0], self.X_em[is_in_cluster, 1], c=cmap(c))

            if c < 0:
                continue

            keywords_idx = np.argsort(np.sum(self.tfidf_matrix[clusters == c, :], axis=0))[
                           -self.n_keywords_static:]
            keywords = reversed(self.tfidf_feature_names[keywords_idx])
            x_avg, y_avg = np.mean(self.X_em[clusters == c, :], axis=0)
            plt.text(x_avg, y_avg, '\n'.join(keywords), ha='center', va='center', bbox=dict(facecolor=cmap(c), alpha=0.2), color='black', fontweight='bold')

    def plot_dynamic(self, x, y, r):
        """
        Plots dynamic labels for the given lens parameters.
        :param x: x coordinate of the lens.
        :param y: y coordinate of the lens.
        :param r: radius of the lens.
        """
        pass


class LemmaTokenizer(object):
    """ Utility class for including lemmatization and stop word removal in tokenization. """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in STOP_WORDS]
