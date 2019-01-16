from lib.base_explorer import BaseExplorer
from scipy.stats import ttest_ind
import numpy as np


class TabExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of tabular dataset projection. """

    def __init__(self, p_threshold=0.01, max_static_labels=3, max_dynamic_labels=5, fig_size=(12, 10)):
        """
        :param p_threshold: statistical significance (p-value) threshold for annotations
        :param max_static_labels: maximum number of static labels per cluster.
        :param max_dynamic_labels: maximum number of dynamic labels (shown for selected group of examples)
        :param fig_size: Figure size.
        """

        self.p_threshold = p_threshold
        self.df_numeric = None
        self.df_discrete = None

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

        self.df_numeric = df.select_dtypes(include=[np.number]).astype(np.float32)
        self.df_discrete = df.select_dtypes(include=[object, 'category', 'bool']).astype(object)

    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        # t-test for continuous attributes
        numeric_in_cluster = self.df_numeric.loc[is_in_cluster].values
        numeric_out_cluster = self.df_numeric.loc[~is_in_cluster].values
        # TODO: we only have to compute means and stds once and use the other ttest call
        _, p_values = ttest_ind(numeric_in_cluster, numeric_out_cluster, equal_var=False)

        p_order = np.argsort(p_values)
        threshold_idx = np.searchsorted(p_values, self.p_threshold, sorter=p_order, side='right')
        selected_idx = p_order[:min(threshold_idx, max_labels)]

        # generate labels
        selected_columns = self.df_numeric.columns.values[selected_idx]
        avg_in_cluster = np.mean(numeric_in_cluster[:, selected_idx], axis=0)
        avg_out_cluster = np.mean(numeric_out_cluster[:, selected_idx], axis=0)

        return ['{} = {:.2f} ({})'.format(c, avg_in, 'high' if avg_in > avg_out else 'low')
                for c, avg_in, avg_out in zip(selected_columns, avg_in_cluster, avg_out_cluster)]
