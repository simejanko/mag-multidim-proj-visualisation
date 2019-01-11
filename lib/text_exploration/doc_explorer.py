from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import re
from lib.base_explorer import BaseExplorer

NON_ALPHABETIC_REGEX = re.compile('[^a-zA-Z]')
STOP_WORDS = set(stopwords.words('english'))


def remove_non_alphabetic(text):
    return NON_ALPHABETIC_REGEX.sub(' ', text).lower()


class DocExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of documents. """

    def __init__(self, method='tfidf', max_static_labels=3,
                 max_dynamic_labels=5, fig_size=(12, 10)):
        """
        :param method: Method to use for keyword extraction. Either 'tfidf' or 'g2'
        :param max_static_labels: Number of keywords to display per cluster
        :param max_dynamic_labels: Number of keywords to display for lense exploration
        :param fig_size: Figure size.
        """
        self.method = method

        self.tf_matrix = None
        self.tf_feature_names = None
        self.tf_totals_words = None
        self.tf_totals_documents = None
        self.tf_expected = None

        super().__init__(max_static_labels=max_static_labels, max_dynamic_labels=max_dynamic_labels, fig_size=fig_size)

    def fit(self, docs, X_em, clusters, ax=None):
        """
        Performs text preprocessing and feature extraction that's needed for keyword extraction. Remembers what is needed for lens exploration.
        :param docs: list of text documents (strings)
        :param X_em: numpy array of embeddings with shape (n_samples, 2)
        :param clusters: numpy array of cluster labels with shape (n_samples,)
        :param ax: specify existing matplotlib axis to use for this plot.
        """

        super().fit(docs, X_em, clusters, ax=ax)

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

    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        extract_keywords = self._keywords_tfidf if self.method == 'tfidf' else self._keywords_g2
        return extract_keywords(is_in_cluster, max_labels)

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


class LemmaTokenizer(object):
    """ Utility class for including lemmatization and stop word removal in tokenization. """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in STOP_WORDS]
