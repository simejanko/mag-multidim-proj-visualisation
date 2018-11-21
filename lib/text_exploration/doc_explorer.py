from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib import pyplot as plt
import re

NON_ALPHABETIC_REGEX = re.compile('[^a-zA-Z]')


class DocExplorer():
    """ Visualisation tool for static and dynamic exploration of documents. """

    def __init__(self, embedding=TSNE(), clustering=DBSCAN(), n_keywords_static=3, n_keywords_dynamic=5):
        """
        :param embedding: Embedding object with sklearn's interface (fit_transform method)
        :param clustering: Clustering object with sklearn's interface (fit_predict method)
        :param n_keywords_static: Number of keywords to display per cluster
        :param n_keywords_dynamic: Number of keywords to display for lense exploration
        """
        self.embedding = embedding
        self.clustering = clustering
        self.n_keywords_static = n_keywords_static
        self.n_keywords_dynamic = n_keywords_dynamic

        self.tfidf_matrix = None
        self.tfidf_feature_names = None

        self.X_em = None

    def fit(self, docs):
        """
        Performs text preprocessing, feature extraction and embedding. Remembers what is needed for lens exploration.
        :param docs: list of text documents (strings)
        """

        def remove_non_alphabetic(text):
            return NON_ALPHABETIC_REGEX.sub(' ', text).lower()

        tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer(),
                                           preprocessor=remove_non_alphabetic)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(docs).toarray()
        self.tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())

        self.X_em = self.embedding.fit_transform(self.tfidf_matrix)

    def plot_static(self):
        """
        Plots static labels for clusters of the embedding.
        """
        clusters = self.clustering.fit_predict(self.X_em)

        plt.scatter(self.X_em[:, 0], self.X_em[:, 1], c=clusters)
        for c in np.unique(clusters):
            keywords_idx = np.argsort(np.sum(self.tfidf_matrix[self.clustering == c, :], axis=0))[
                           -self.n_keywords_static:]
            keywords = reversed(self.tfidf_feature_names[keywords_idx])
            x_avg, y_avg = np.mean(self.X_em[clusters == c, :], axis=0)
            plt.text(x_avg, y_avg, '\n'.join(keywords))

        plt.show()

    def plot_dynamic(self, x, y, r):
        """
        Plots dynamic labels for the given lens parameters.
        :param x: x coordinate of the lens.
        :param y: y coordinate of the lens.
        :param r: radius of the lens.
        """
        pass


class LemmaTokenizer(object):
    """ Utility class for including lemmatization in sklearn's feature extractors. """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
