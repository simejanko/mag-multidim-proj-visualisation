from lib.base_explorer import BaseExplorer
from scipy.stats import ttest_ind
import numpy as np
from lib.utils.statistics import Hypergeometric, p_adjust_bh
import pandas as pd


# TODO: one way to make these methods more efficient is to take into account the fact that we're dynamically exploring
# nearby points and not just any points. One could use structures like kd-trees or similiar and store relevant
# aggregates in the nodes.

# TODO : handle cases when Dataframe only contains numeric or discrete data.
# TODO : try one-sample t-test or t-test for all

class TabExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of tabular dataset projection. """

    def __init__(self, p_threshold=0.01, representative_threshold=0.5, fdr_correction=True, max_static_labels=3,
                 max_dynamic_labels=5, min_cluster_size=5, max_discrete_values=3, fig_size=(12, 10)):
        """
        :param p_threshold: statistical significance (p-value) threshold for annotations
        :param representative_threshold: only show labels that represent at least this proportion of samples in a group.
                                         Addresses the representativeness vs significance problem.
        :param fdr_correction: whether to apply FDR correction to p-values (multiple comparisons problem)
        :param max_static_labels: maximum number of static labels per cluster
        :param max_dynamic_labels: maximum number of dynamic labels (shown for selected group of examples)
        :param max_discrete_values: maximum number of values shown for discrete attributes
        :param fig_size: Figure size
        """

        self.p_threshold = p_threshold
        self.representative_threshold = representative_threshold
        self.fdr_correction = fdr_correction
        self.max_discrete_values = max_discrete_values
        self.df_numeric = None
        self.df_discrete = None
        self.discrete_columns = None
        self.discrete_value_counts = None
        self.hypergeom = None

        super().__init__(max_static_labels=max_static_labels, max_dynamic_labels=max_dynamic_labels,
                         min_cluster_size=min_cluster_size, fig_size=fig_size)

    def fit(self, df, X_em, clusters):
        """
        Performs any kind of preprocessing and caching needed for lens exploration.
        :param df: pandas DataFrame. Only the following dtypes will be considered: (object, category, bool, intXX, floatXX)
        :param X_em: numpy array of embeddings with shape (n_samples, 2)
        :param clusters: numpy array of cluster labels with shape (n_samples,)
        """

        super().fit(df, X_em, clusters)

        self.df_numeric = df.select_dtypes(include=[np.number]).astype(np.float32)

        df_discrete = df.select_dtypes(include=[object, 'category', 'bool']).astype(object)
        self.df_discrete = pd.concat([pd.get_dummies(df_discrete[col]) for col in df_discrete], axis=1,
                                     keys=df_discrete.columns)
        self.hypergeom = Hypergeometric(max=df.shape[0])

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

        # use MultiIndex for better compatibility with discrete p-values Series
        return pd.Series(numeric_p_values,
                         index=pd.MultiIndex.from_tuples(list(zip(self.df_numeric.columns, self.df_numeric.columns))))

    def _discrete_p_values(self, is_in_cluster):
        """
        Computes p-values for discrete attributes using hypergeometric test for every possible discrete value.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :return  pandas Series with p-values
        """
        total_size = self.df_discrete.shape[0]
        cluster_size = np.sum(is_in_cluster)
        discrete_in_cluster_counts = self.df_discrete.loc[is_in_cluster].sum()
        discrete_p_values = [self.hypergeom.p_value(dc, total_size, vc, cluster_size)
                             for vc, dc in zip(self.discrete_value_counts, discrete_in_cluster_counts)]
        discrete_p_values = pd.Series(discrete_p_values, index=discrete_in_cluster_counts.index)

        return discrete_p_values

    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        numeric_p_values = self._numeric_p_values(is_in_cluster)
        discrete_p_values = self._discrete_p_values(is_in_cluster)
        p_values = pd.concat([numeric_p_values, discrete_p_values])

        if self.fdr_correction:
            p_values = p_adjust_bh(p_values)

        # pick min values for discrete attributes
        p_values = p_values.groupby(level=0).min().sort_values()

        # apply p-value threshold
        threshold_idx = np.searchsorted(p_values.values, self.p_threshold, side='right')
        selected_columns = p_values[:threshold_idx].index.values

        # generate labels
        labels = []
        for c in selected_columns:
            if c in self.df_numeric.columns:
                # label for continuous attribute
                #TODO: probably inefficient
                s_in = self.df_numeric.loc[is_in_cluster, c]
                avg_in = s_in.mean()
                avg_out = self.df_numeric.loc[~is_in_cluster, c].mean()

                #representative proportion for this label and group
                rep_proportion = ((s_in > avg_out) if avg_in > avg_out else (s_in < avg_out)).sum()/s_in.size

                label = '{} = {:.2f} ({})'.format(c, avg_in, 'high' if avg_in > avg_out else 'low')
            else:
                # label for discrete attribute
                pv = discrete_p_values[c]
                selected_values = pv[pv <= self.p_threshold].sort_values()[:self.max_discrete_values].index.tolist()
                # TODO: probably inefficient
                s_in = self.df_discrete.loc[is_in_cluster, (c, selected_values)].any(axis=1)
                rep_proportion = s_in.sum()/s_in.size

                selected_values = list(map(str, selected_values))
                if len(selected_values) > 1:
                    selected_values[-1] = ' or ' + selected_values[-1]

                label = '{} = {}'.format(c, ', '.join(selected_values[:-1]) + selected_values[-1])

            if rep_proportion >= self.representative_threshold:
                labels.append(label)

            if len(labels) == max_labels:
                break

        return labels
