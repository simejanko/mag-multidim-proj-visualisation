from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib import pyplot as plt
import re
from sklearn.neighbors import KDTree

NON_ALPHABETIC_REGEX = re.compile('[^a-zA-Z]')
STOP_WORDS = set(stopwords.words('english'))


def remove_non_alphabetic(text):
    return NON_ALPHABETIC_REGEX.sub(' ', text).lower()


class DocExplorer():
    """ Visualisation tool for static and dynamic exploration of documents. """

    STAT_FONT_SIZE_MAX = 16
    STAT_FONT_SIZE_MIN = 9
    DYNM_FONT_SIZE = 12

    def __init__(self, method='tfidf', n_keywords_static=3,
                 n_keywords_dynamic=5, fig_size=(12, 10)):
        """
        :param method: Method to use for keyword extraction. Either 'tfidf' or 'g2'
        :param n_keywords_static: Number of keywords to display per cluster
        :param n_keywords_dynamic: Number of keywords to display for lense exploration
        :param fig_size: Figure size.
        """
        self.method = method
        self.extract_keywords = self._keywords_tfidf if self.method == 'tfidf' else self._keywords_g2
        self.n_keywords_static = n_keywords_static
        self.n_keywords_dynamic = n_keywords_dynamic

        self.fig, self.ax = plt.subplots(figsize=fig_size)
        self.scatter_plot = None
        self.lens = plt.Circle((0, 0), 0, edgecolor='black', fill=False)
        self.ax.add_artist(self.lens)
        self.annotations = []

        self.tf_matrix = None
        self.tf_feature_names = None

        self.X_em = None
        self.clusters = None
        self.kd_tree = None

        self.tf_totals_words = None
        self.tf_totals_documents = None
        self.tf_expected = None

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

        # G2 method specific caching
        if self.method == 'g2':
            self.tf_totals_words = self.tf_matrix.sum(axis=0)
            self.tf_totals_documents = self.tf_matrix.sum(axis=1)
            self.tf_expected = self.tf_totals_words / self.tf_totals_words.sum()

        if X_em is None:
            self.X_em = TSNE().fit_transform(self.tf_matrix)
        else:
            self.X_em = X_em

        if clusters is None:
            self.clusters = DBSCAN().fit_predict(self.tf_matrix)
        else:
            self.clusters = clusters

        self.kd_tree = KDTree(self.X_em, leaf_size=20)

    def _keywords_tfidf(self, is_in_cluster, n_keywords):
        """
        Get keywords for a given cluster using tf-idf method.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param n_keywords: number of keywords to extract.
        :return: list of keywords
        """
        keywords_idx = np.argsort(np.sum(self.tf_matrix[is_in_cluster, :], axis=0))[
                       -n_keywords:]
        return list(reversed(self.tf_feature_names[keywords_idx]))

    def _keywords_g2(self, is_in_cluster, n_keywords):
        """
        Get keywords for a given cluster using g2 method.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param n_keywords: number of keywords to extract.
        :return: list of keywords
        """
        expected_in_cluster = self.tf_totals_documents[is_in_cluster].sum() * self.tf_expected
        expected_out_cluster = self.tf_totals_documents[~is_in_cluster].sum() * self.tf_expected
        tf_in_cluster = self.tf_matrix[is_in_cluster, :].sum(axis=0)
        tf_out_cluster = self.tf_totals_words - tf_in_cluster

        g2 = 2 * (tf_in_cluster * np.log((1e-10 + tf_in_cluster) / expected_in_cluster) +
                  (tf_out_cluster * np.log((1e-10 + tf_out_cluster) / expected_out_cluster)))

        keywords_idx = np.argsort(g2)[-n_keywords:]
        return list(reversed(self.tf_feature_names[keywords_idx]))

    def plot_static(self):
        """
        Plots static labels for clusters of the embedding.
        """
        self.scatter_plot = self.ax.scatter(self.X_em[:, 0], self.X_em[:, 1], facecolor=(0.5, 0.5, 0.5, 1),
                                            edgecolor=(0, 0, 0, 1))
        self.ax.axis('equal')

        for c in np.unique(self.clusters):
            if c < 0:
                continue

            is_in_cluster = self.clusters == c
            keywords = self.extract_keywords(is_in_cluster, self.n_keywords_static)

            x_avg, y_avg = np.mean(self.X_em[is_in_cluster, :], axis=0)
            font_sizes = np.linspace(self.STAT_FONT_SIZE_MAX, self.STAT_FONT_SIZE_MIN, num=self.n_keywords_static)
            dy = [0, 0]
            for i, keyword in enumerate(keywords):
                annotation_side = (i % 2 * 2 - 1)
                dy[i % 2] += annotation_side * font_sizes[i] * 0.6
                self.ax.annotate(keyword, (x_avg, y_avg), (0, dy[i % 2]), textcoords='offset points', ha='center',
                                 va='center', fontsize=font_sizes[i],
                                 bbox=dict(boxstyle='square,pad=0.1', facecolor='red', alpha=0.5, linewidth=0),
                                 color='white', fontweight='bold')
                dy[i % 2] += annotation_side * font_sizes[i] * 0.6
        return self.ax

    def plot_dynamic(self, x, y, r):
        """
        Updates dynamic labels for the given lens parameters.
        :param x: x coordinate of the lens.
        :param y: y coordinate of the lens.
        :param r: radius of the lens.
        """
        self.lens.center = x, y
        self.lens.set_radius(r)

        # remove old dynamic annotations
        for ann in self.annotations:
            ann.remove()
        self.annotations = []

        is_selected = np.zeros(self.tf_matrix.shape[0], bool)
        selected_idx = self.kd_tree.query_radius([[x, y]], r=r)[0]
        is_selected[selected_idx] = True
        if np.sum(is_selected) == 0:
            return self.ax

        keywords = self.extract_keywords(is_selected, self.n_keywords_dynamic)

        face_colors = [(0.5, 0.5, 0.5, 1) if s else (0.5, 0.5, 0.5, 0.4) for s in is_selected]
        edge_colors = [(0, 0, 0, 1) if s else (0, 0, 0, 0.4) for s in is_selected]
        self.scatter_plot.set_facecolor(face_colors)
        self.scatter_plot.set_edgecolor(edge_colors)

        dy = 1.2 * self.DYNM_FONT_SIZE * self.n_keywords_dynamic / 2
        for i, keyword in enumerate(keywords):
            ann = self.ax.annotate(keyword, (x - r, y), (0, dy), textcoords='offset points', ha='right',
                                   va='center', fontsize=self.DYNM_FONT_SIZE,
                                   bbox=dict(boxstyle='square,pad=0.1', fill=False),
                                   color='black')
            self.annotations.append(ann)
            dy -= 1.2 * self.DYNM_FONT_SIZE

        return self.ax

    def plot_interactive(self, r=5):
        """
        Connect click event with dynamic label updating for interactivity.
        :param r: lens radius
        """

        def onclick(event):
            if event.button != 1:
                return

            self.plot_dynamic(event.xdata, event.ydata, self.lens.get_radius())

        self.lens.set_radius(r)
        self.fig.canvas.mpl_connect('button_press_event', onclick)


class LemmaTokenizer(object):
    """ Utility class for including lemmatization and stop word removal in tokenization. """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in STOP_WORDS]
