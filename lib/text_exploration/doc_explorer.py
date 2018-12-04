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


def remove_non_alphabetic(text):
    return NON_ALPHABETIC_REGEX.sub(' ', text).lower()


class DocExplorer():
    """ Visualisation tool for static and dynamic exploration of documents. """

    def __init__(self, method='tfidf', n_keywords_static=3,
                 n_keywords_dynamic=5):
        """
        :param method: Method to use for keyword extraction. Either 'tfidf' or 'g2'
        :param n_keywords_static: Number of keywords to display per cluster
        :param n_keywords_dynamic: Number of keywords to display for lense exploration
        """
        self.method = method
        self.n_keywords_static = n_keywords_static
        self.n_keywords_dynamic = n_keywords_dynamic

        self.tf_matrix = None
        self.tf_feature_names = None

        self.X_em = None
        self.clusters = None

    def fit(self, docs, X_em=None, clusters=None):
        """
        Performs text preprocessing and feature extraction that's needed for keyword extraction. Remembers what is needed for lens exploration.
        :param docs: list of text documents (strings)
        :param X_em: numpy array of embeddings with shape (n_samples, 2). If None, t-SNE with default parameters
         is used on tf or tf-idf matrix depending on the keyword extraction method used.
        :param clusters: numpy array of cluster labels with shape (n_samples,). If None, DBSCAN with default parameters
        is used on tf or tf-idf matrix depending on the keyword extraction method used.
        """

        tf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                        preprocessor=remove_non_alphabetic,
                                        max_df=0.5, use_idf=self.method == 'tfidf')
        self.tf_matrix = tf_vectorizer.fit_transform(docs).toarray()
        self.tf_feature_names = np.array(tf_vectorizer.get_feature_names())

        if X_em is None:
            self.X_em = TSNE().fit_transform(self.tf_matrix)
        else:
            self.X_em = X_em

        if clusters is None:
            self.clusters = DBSCAN().fit_predict(self.tf_matrix)
        else:
            self.clusters = clusters

    def _keywords_tfidf(self, clusters):
        """
        :return: Keywords for each cluster using tf-idf method.
        """
        keywords = dict()
        for c in np.unique(clusters):
            keywords_idx = np.argsort(np.sum(self.tf_matrix[clusters == c, :], axis=0))[
                           -self.n_keywords_static:]
            keywords[c] = list(reversed(self.tf_feature_names[keywords_idx]))
        return keywords

    def _keywords_g2(self, clusters):
        """
        :return: Keywords for each cluster using G2 method.
        """
        tf_totals_words = self.tf_matrix.sum(axis=0)
        tf_totals_documents = self.tf_matrix.sum(axis=1)
        tf_expected = tf_totals_words / tf_totals_words.sum()

        keywords = dict()
        for c in np.unique(clusters):
            is_in_cluster = clusters == c
            expected_in_cluster = tf_totals_documents[is_in_cluster].sum() * tf_expected
            expected_out_cluster = tf_totals_documents[~is_in_cluster].sum() * tf_expected
            tf_in_cluster = self.tf_matrix[is_in_cluster, :].sum(axis=0)
            tf_out_cluster = tf_totals_words - tf_in_cluster

            g2 = 2 * (tf_in_cluster * np.log((1e-10 + tf_in_cluster) / expected_in_cluster) +
                      (tf_out_cluster * np.log((1e-10 + tf_out_cluster) / expected_out_cluster)))

            keywords_idx = np.argsort(g2)[-self.n_keywords_static:]
            keywords[c] = list(reversed(self.tf_feature_names[keywords_idx]))
        return keywords

    def plot_static(self, fig_size=(12, 10)):
        """
        Plots static labels for clusters of the embedding.
        :param size: figure size
        """
        extract_keywords = self._keywords_tfidf if self.method == 'tfidf' else self._keywords_g2
        keywords = extract_keywords(self.clusters)

        f, ax = plt.subplots(figsize=fig_size)
        ax.scatter(self.X_em[:, 0], self.X_em[:, 1], c='gray', edgecolors='black')

        for c in np.unique(self.clusters):
            if c < 0:
                continue

            x_avg, y_avg = np.mean(self.X_em[self.clusters == c, :], axis=0)
            font_sizes = np.linspace(16, 9, num=self.n_keywords_static)
            dy = [0, 0]
            for i, keyword in enumerate(keywords[c]):
                annotation_side = (i % 2 * 2 - 1)
                dy[i % 2] += annotation_side * font_sizes[i] * 0.6
                ax.annotate(keyword, (x_avg, y_avg), (0, dy[i % 2]), textcoords='offset points', ha='center',
                            va='center', fontsize=font_sizes[i],
                            bbox=dict(boxstyle='square,pad=0.1', facecolor='red', alpha=0.5, linewidth=0),
                            color='white', fontweight='bold')
                dy[i % 2] += annotation_side * font_sizes[i] * 0.6
        return ax

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
