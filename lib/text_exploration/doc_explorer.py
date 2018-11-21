from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

class DocExplorer():
    """ Visualisation tool for static and dynamic exploration of documents. """

    def __init__(self, embedding=TSNE(), clustering=DBSCAN()):
        """
        :param embedding: Embedding object with sklearn's interface (fit_transform method)
        :param clustering: Clustering object with sklearn's interface (fit_predict method)
        """
        self.embedding = embedding
        self.clustering = clustering

        self.tfidf_matrix = None
        self.tfidf_feature_names = None

        self.X_em = None
        self.clusters = None

    def _extract_features(self, docs):
        """
        Preprocesses documents and extracts tf-idf features. Preprocessing includes transforming text to lowercase,
        removing stopwords, tokenization and lemmatization.
        :param docs: list of documents
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
        self.tfidf_feature_names = tfidf_vectorizer.get_feature_names()


    def fit(self, docs):
        """
        Performs text preprocessing, feature extraction, embedding, clustering and keyword extraction.
        :param docs: list of text documents (strings)
        """
        self._extract_features(docs)
        self.X_em = self.embedding.fit_transform(self.tfidf_matrix)
        self.clusters = self.clustering.fit_predict(self.X_em)

        #TODO: keyword extraction per cluster

    def plot_static(self):
        """
        Plots static labels for clusters of the embedding.
        :param docs:
        :return:
        """
        pass

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
