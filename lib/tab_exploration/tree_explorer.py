from lib.base_explorer import BaseExplorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class TreeExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of tabular dataset projection using decision trees. """

    def __init__(self, max_static_labels=3, max_dynamic_labels=5, min_cluster_size=5, fig_size=(12, 10)):
        """
        :param max_static_labels: maximum number of static labels per cluster
        :param max_dynamic_labels: maximum number of dynamic labels (shown for selected group of examples)
        :param fig_size: Figure size.
        """
        super().__init__(max_static_labels=max_static_labels, max_dynamic_labels=max_dynamic_labels,
                         min_cluster_size=min_cluster_size, fig_size=fig_size)

        self.label_encoders = None
        self.df = None

    def fit(self, df, X_em, clusters):
        """
        Performs any kind of preprocessing and caching needed for lens exploration.
        :param df: pandas DataFrame. Only the following dtypes will be considered: (object, category, bool, intXX, floatXX)
        :param X_em: numpy array of embeddings with shape (n_samples, 2)
        :param clusters: numpy array of cluster labels with shape (n_samples,)
        """

        super().fit(df, X_em, clusters)

        self.df = df.copy()

        self.label_encoders = {}
        discrete_columns = self.df.columns[self.df.dtypes.isin([object, 'category', 'bool'])]
        for c in discrete_columns:
            le = LabelEncoder()
            self.df[c] = le.fit_transform(self.df[c].values)
            self.label_encoders[c] = le


    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        dt = DecisionTreeClassifier(max_depth=1)
        dt.fit(self.df.values, is_in_cluster)
        
        return []
