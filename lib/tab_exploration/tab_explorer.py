from lib.base_explorer import BaseExplorer
from scipy.stats import ttest_ind
import numpy as np
from lib.utils.statistics import Hypergeometric
import pandas as pd


# TODO: indicate private members

# TODO: one way to make these methods more efficient is to take into account the fact that we're dynamically exploring
# nearby points and not just any points. One could use structures like kd-trees or similiar and store relevant
# aggregates in the nodes.

class TabExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of tabular dataset projection. """

    def __init__(self, p_threshold=0.01, max_static_labels=3, max_dynamic_labels=5, max_discrete_values=3,
                 fig_size=(12, 10)):
        """
        :param p_threshold: statistical significance (p-value) threshold for annotations
        :param max_static_labels: maximum number of static labels per cluster
        :param max_dynamic_labels: maximum number of dynamic labels (shown for selected group of examples)
        :param max_discrete_values: maximum number of values shown for discrete attributes
        :param fig_size: Figure size.
        """

        self.p_threshold = p_threshold
        self.max_discrete_values = max_discrete_values
        self.df_numeric = None
        self.df_discrete = None
        self.discrete_columns = None
        self.discrete_value_counts = None
        self.hypergeom = None

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

        df_discrete = df.select_dtypes(include=[object, 'category', 'bool']).astype(object)
        self.df_discrete = pd.concat([pd.get_dummies(df_discrete[col]) for col in df_discrete], axis=1,
                                     keys=df_discrete.columns)
        # TODO: max can be max(largest_cluster, N-smallest cluster) in case out-cluster K and N values turn out to be correct approach (Wikipedia notation)
        self.hypergeom = Hypergeometric(max=df.shape[0])

        # TODO: remove, in case out-cluster K and N values turn out to be correct approach (Wikipedia notation)
        self.discrete_value_counts = self.df_discrete.sum()

    def _numeric_p_values(self, is_in_cluster):
        """
        Computes p-values for continuous attributes using t-test between in-cluster and out-cluster groups.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :return pandas Series with p-values
        """
        numeric_in_cluster = self.df_numeric.loc[is_in_cluster]
        numeric_out_cluster = self.df_numeric.loc[~is_in_cluster]
        # TODO: we only have to compute means and stds once and use the other ttest call
        _, numeric_p_values = ttest_ind(numeric_in_cluster.values, numeric_out_cluster.values, equal_var=False)

        return pd.Series(numeric_p_values, index=self.df_numeric.columns)

    def _discrete_p_values(self, is_in_cluster):
        """
        Computes p-values for discrete attributes using hypergeometric test for every possible discrete value.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :return  tuple with two pandas Series: (p-values for all discrete values, minimum p-values per discrete attribute)
        """
        total_size = self.df_discrete.shape[0]
        cluster_size = np.sum(is_in_cluster)
        # TODO : possible bias: We used in vs out cluster for t-test, but for hypergeom we're using in-cluster for k and n
        #       and global values for K and N... should we use out-cluster values for K and N? (Wikipedia notation)
        discrete_in_cluster_counts = self.df_discrete.loc[is_in_cluster].sum()
        discrete_p_values = self.discrete_value_counts.combine(discrete_in_cluster_counts,
                                                               lambda vc, dc: self.hypergeom.p_value(dc, total_size, vc,
                                                                                                     cluster_size))
        discrete_min_p_values = discrete_p_values.groupby(level=0).min()

        return discrete_p_values, discrete_min_p_values

    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        numeric_p_values = self._numeric_p_values(is_in_cluster)
        discrete_p_values, discrete_min_p_values = self._discrete_p_values(is_in_cluster)

        # select attributes to display
        p_values = pd.concat([numeric_p_values, discrete_min_p_values]).sort_values()
        threshold_idx = np.searchsorted(p_values.values, self.p_threshold, side='right')
        selected_columns = p_values[:min(max_labels, threshold_idx)].index.values

        # generate labels
        labels = []
        for c in selected_columns:
            if c in self.df_numeric.columns:
                # label for continuous attribute
                avg_in = self.df_numeric[c, is_in_cluster].mean()
                avg_out = self.df_numeric[c, ~is_in_cluster].mean()

                label = '{} = {:.2f} ({})'.format(c, avg_in, 'high' if avg_in > avg_out else 'low')
            else:
                # label for discrete attribute
                pv = discrete_p_values[c]
                selected_values = pv[pv <= self.p_threshold][:self.max_discrete_values].index.tolist()
                selected_values = list(map(str, selected_values))
                if len(selected_values) > 1:
                    selected_values[-1] = ' or ' + selected_values[-1]

                label = '{} = {}'.format(c, ', '.join(selected_values[:-1]) + selected_values[-1])

            labels.append(label)

        return labels
