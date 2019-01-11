from lib.base_explorer import BaseExplorer


class TabExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of tabular dataset projection. """

    def __init__(self, p_threshold=0.1, max_static_labels=3, max_dynamic_labels=5, fig_size=(12, 10)):
        """
        :param p_threshold: statistical significance (p-value) threshold for annotations
        :param max_static_labels: maximum number of static labels per cluster.
        :param max_dynamic_labels: maximum number of dynamic labels (shown for selected group of examples)
        :param fig_size: Figure size.
        """

        self.p_threshold = p_threshold
        self.df = None

        super().__init__(max_static_labels=max_static_labels, max_dynamic_labels=max_dynamic_labels, fig_size=fig_size)

    def fit(self, df, X_em, clusters, ax=None):
        """
        Performs any kind of preprocessing and caching needed for lens exploration.
        :param df: pandas DataFrame. Only the following dtypes will be considered: (object, category, bool, intXX, floatXX)
        :param X_em: numpy array of embeddings with shape (n_samples, 2)
        :param clusters: numpy array of cluster labels with shape (n_samples,)
        :param ax: specify existing matplotlib axis to use for this visualisation
        """

        super().fit(df, X_em, clusters, ax=ax)

        self.df = df

    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        return []
