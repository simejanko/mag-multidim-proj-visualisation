from lib.base_explorer import BaseExplorer
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np
from lib.utils.statistics import Hypergeometric, p_adjust_bh, two_tailed_p_to_one_tailed
import pandas as pd

#TODO: use normal indexing instead of bool indexing... a bit faster

# TODO: one way to make these methods more efficient is to take into account the fact that we're dynamically exploring
# nearby points and not just any points. One could use structures like kd-trees or similiar and store relevant
# aggregates in the nodes.


class TabExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of tabular dataset projection. """

    def __init__(self, p_threshold=0.01, representative_threshold=0.5, use_two_sample_test=False, one_tailed_test=False,
                 fdr_correction=True, max_static_labels=3, max_dynamic_labels=5, min_cluster_size=5, max_discrete_values=3,
                 fig_size=(12, 10)):
        """
        :param p_threshold: statistical significance (p-value) threshold for annotations
        :param representative_threshold: only show labels that represent at least this proportion of samples in a group.
                                         Addresses the representativeness vs significance problem.
        :param use_two_sample_test: whether to use two sample tests (t-test) or one sample tests
                                    (hypergeometric test for discrete values and t-test for continuous values)
        :param one_tailed_test: whether to use one tailed test for continuous attributes (it's always used for discrete attributes)
        :param fdr_correction: whether to apply FDR correction to p-values (multiple comparisons problem)
        :param max_static_labels: maximum number of static labels per cluster
        :param max_dynamic_labels: maximum number of dynamic labels (shown for group of examples selected with a lens)
        :param max_discrete_values: maximum number of different values shown per discrete attribute
        :param fig_size: Figure size
        """

        self.p_threshold = p_threshold
        self.representative_threshold = representative_threshold
        self.use_two_sample_test = use_two_sample_test
        self.one_tailed_test = one_tailed_test
        self.fdr_correction = fdr_correction
        self.max_discrete_values = max_discrete_values

        self.df_numeric = None
        self.df_discrete = None
        self.numeric_means = None
        self.discrete_columns = None
        self.discrete_value_counts = None
        self.discrete_value_means = None
        self.hypergeom = None
        self.labels_dict = None

        super().__init__(max_static_labels=max_static_labels, max_dynamic_labels=max_dynamic_labels,
                         min_cluster_size=min_cluster_size, fig_size=fig_size)

    def fit(self, df, X_em, clusters, labels_dict={}):
        """
        Performs any kind of preprocessing and caching needed for lens exploration.

        :param df: pandas DataFrame. Only the following dtypes will be considered: (object, category, bool, intXX, floatXX)
        :param X_em: numpy array of embeddings with shape (n_samples, 2)
        :param clusters: numpy array of cluster labels with shape (n_samples,)
        :param labels_dict: dict mapping from attribute names to labels to be displayed in case attribute is deemed significant
                            (this overrides default way of displaying labels)
        """

        super().fit(df, X_em, clusters)

        self.df_numeric = df.select_dtypes(include=[np.number]).astype(np.float32)
        self.numeric_means = self.df_numeric.mean()

        self.df_discrete = df.select_dtypes(include=[object, 'category', 'bool']).astype(object)
        if not self.df_discrete.empty:
            self.df_discrete = pd.concat([pd.get_dummies(self.df_discrete[col]) for col in self.df_discrete], axis=1,
                                         keys=self.df_discrete.columns)
            self.discrete_value_counts = self.df_discrete.sum()
            self.discrete_value_means = self.df_discrete.mean()

        self.hypergeom = Hypergeometric(max=df.shape[0])

        self.labels_dict = labels_dict

    def _numeric_p_values(self, is_in_cluster):
        """
        Computes p-values for continuous attributes using t-test between in-cluster and out-cluster groups.

        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :return pandas Series with p-values
        """
        if self.df_numeric.empty:
            return None

        numeric_in_cluster = self.df_numeric.loc[is_in_cluster]
        # TODO: we only have to compute means and stds once and use the other ttest call
        if self.use_two_sample_test:
            numeric_out_cluster = self.df_numeric.loc[~is_in_cluster]
            ts, numeric_p_values = ttest_ind(numeric_in_cluster.values, numeric_out_cluster.values, equal_var=False)
        else:
            ts, numeric_p_values = ttest_1samp(numeric_in_cluster.values, self.numeric_means.values)

        if self.one_tailed_test:
            numeric_p_values = two_tailed_p_to_one_tailed(ts, numeric_p_values)

        # use MultiIndex for better compatibility with discrete p-values Series
        return pd.Series(numeric_p_values,
                         index=pd.MultiIndex.from_tuples(list(zip(self.df_numeric.columns, self.df_numeric.columns))))

    def _discrete_p_values(self, is_in_cluster):
        """
        Computes p-values for discrete attributes using hypergeometric test for every possible discrete value.

        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :return  pandas Series with p-values
        """
        if self.df_discrete.empty:
            return None

        total_size = self.df_discrete.shape[0]
        discrete_in_cluster = self.df_discrete.loc[is_in_cluster]

        if self.use_two_sample_test:
            discrete_out_cluster = self.df_discrete.loc[~is_in_cluster]
            ts, discrete_p_values = ttest_ind(discrete_in_cluster.values, discrete_out_cluster.values, equal_var=False)
            #convert to one-tailed t-test p-values since we're only looking for over-representations
            discrete_p_values = two_tailed_p_to_one_tailed(ts, discrete_p_values)
        else:
            discrete_in_cluster_counts = discrete_in_cluster.sum()
            cluster_size = np.sum(is_in_cluster)
            discrete_p_values = [self.hypergeom.p_value(dc, total_size, vc, cluster_size)
                                 for vc, dc in zip(self.discrete_value_counts, discrete_in_cluster_counts)]

        return pd.Series(discrete_p_values, index=self.df_discrete.columns)

    #TODO: refactor
    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.

        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        numeric_p_values = self._numeric_p_values(is_in_cluster)
        discrete_p_values = self._discrete_p_values(is_in_cluster)

        p_values = pd.concat([pv for pv in (numeric_p_values, discrete_p_values) if pv is not None])

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
                # TODO: probably inefficient
                s_in = self.df_numeric.loc[is_in_cluster, c]
                avg_in = s_in.mean()
                if self.use_two_sample_test:
                    avg_out = self.df_numeric.loc[~is_in_cluster, c].mean()
                else:
                    avg_out = self.numeric_means[c]

                rep_proportion = ((s_in > avg_out) if avg_in > avg_out else (s_in < avg_out)).sum() / s_in.size

                label = '{} = {:.2f} ({})'.format(c, avg_in, 'high' if avg_in > avg_out else 'low')
            else:
                # label for discrete attribute
                pv = discrete_p_values[c]
                selected_values = pv[pv <= self.p_threshold].sort_values()[:self.max_discrete_values].index.tolist()
                # TODO: probably inefficient
                s_in = self.df_discrete.loc[is_in_cluster, (c, selected_values)].any(axis=1)
                rep_proportion = s_in.sum() / s_in.size

                selected_values = list(map(str, selected_values))
                if len(selected_values) > 1:
                    selected_values[-1] = ' or ' + selected_values[-1]

                label = '{} = {}'.format(c, ', '.join(selected_values[:-1]) + selected_values[-1])

            if c in self.labels_dict:
                label = self.labels_dict[c]

            # does label represent a big enough portion of the samples in a group
            if rep_proportion >= self.representative_threshold:
                labels.append(label)

            if len(labels) == max_labels:
                break

        return labels


class GiniExplorer(BaseExplorer):
    """ Visualisation tool for static and dynamic exploration of tabular dataset projection using information measure (Gini impurity). """

    def __init__(self, g_threshold=0.1, n_sample=50, find_intervals=True, max_static_labels=3,
                 max_dynamic_labels=5, min_cluster_size=5,
                 max_discrete_values=3, fig_size=(12, 10)):
        """
        :param g_threshold: relative gini gain threshold for labels. Gini gain is limited between 0 (worst)
               and total gini impurity for a given cluster split (best). This parameter is relative and
               is independent of number & sizes of clusters so it should be between 0 and 1.
        :param n_sample: number of split points to sample when considering continuous attributes
        :param find_intervals: whether to try and find intervals or only single split points for continuous attributes ()
        :param max_static_labels: maximum number of static labels per cluster
        :param max_dynamic_labels: maximum number of dynamic labels (shown for selected group of examples)
        :param fig_size: Figure size.
        """
        super().__init__(max_static_labels=max_static_labels, max_dynamic_labels=max_dynamic_labels,
                         min_cluster_size=min_cluster_size, fig_size=fig_size)

        self.g_threshold = g_threshold
        self.n_sample = n_sample
        self.find_intervals = find_intervals
        self.max_discrete_values = max_discrete_values
        self.df_numeric = None
        self.numeric_split_sample = None
        self.df_discrete = None

    def fit(self, df, X_em, clusters):
        """
        Performs any kind of preprocessing and caching needed for lens exploration.
        :param df: pandas DataFrame. Only the following dtypes will be considered: (object, category, bool, intXX, floatXX)
        :param X_em: numpy array of embeddings with shape (n_samples, 2)
        :param clusters: numpy array of cluster labels with shape (n_samples,)
        """

        super().fit(df, X_em, clusters)
        self.df_numeric = df.select_dtypes(include=[np.number]).astype(np.float32)
        # sample for split point candidates
        df_numeric_sample = self.df_numeric.sample(n=self.n_sample)
        self.numeric_split_sample = {c: df_numeric_sample[c].unique().tolist() for c in df_numeric_sample.columns}

        self.df_discrete = df.select_dtypes(include=[object, 'category', 'bool']).astype(object)
        if not self.df_discrete.empty:
            self.df_discrete = pd.concat(
                [pd.get_dummies(self.df_discrete[col]).astype(bool) for col in self.df_discrete], axis=1,
                keys=self.df_discrete.columns)

    def _gini_impurity(self, is_class):
        """
        Computes Gini impurity for binary class.
        :param is_class: numpy bool array indicating class memberships
        :return: gini impurity, proportion in class
        """
        p = is_class.sum() / is_class.size
        return 1 - (p ** 2 + (1 - p) ** 2), p

    def _gini_gain(self, bool_atr, is_class):
        """
        Computes Gini gain for binary class and binary atribute split.
        :param bool_atr: numpy bool array indicating binary atribute split
        :param is_class: numpy bool array indicating class memberships
        :return: gini gain, directionality bool
        """
        # TODO: this only needs to be computed once per cluster
        total_impurity, total_p = self._gini_impurity(is_class)

        n_bool_atr = bool_atr.sum()

        # handle cases when we sample extreme values of attribute values
        if n_bool_atr == 0 or n_bool_atr == bool_atr.size:
            return 0, False

        p_atr = n_bool_atr / bool_atr.size
        g1, p1 = self._gini_impurity(is_class[bool_atr])
        g2, _ = self._gini_impurity(is_class[~bool_atr])

        return total_impurity - (p_atr * g1 + (1 - p_atr) * g2), p1 > total_p

    def _discrete_gains(self, is_in_cluster):
        """
        Computes gains for discrete attributes.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :return dict atr_name -> (Gini gain, associated atr_values)
        """
        ret = dict()
        for atr in self.df_discrete.columns.get_level_values(0).unique():
            best_values = []
            best_gain = 0

            df_atr = self.df_discrete[atr]
            remaining_values = set(df_atr.columns)
            prev = np.zeros(is_in_cluster.size).astype(bool)
            for _ in range(self.max_discrete_values):
                value_gains = {v: self._gini_gain(df_atr[v] | prev, is_in_cluster) for v in remaining_values}
                # we're only interested in over representations of discrete values
                value_gains = {v: g for v, (g, d) in value_gains.items() if d}
                if len(value_gains) == 0:
                    break

                best_value = max(value_gains, key=lambda v: value_gains[v])

                if value_gains[best_value] <= best_gain:
                    break

                best_values.append(best_value)
                remaining_values.remove(best_value)
                best_gain = value_gains[best_value]

            if len(best_values) >= 1:
                ret[atr] = (best_gain, best_values)

        return ret

    def _continuous_gains(self, is_in_cluster):
        """
        Computes gains for continuous attributes.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :return dict atr_name -> (Gini gain, associated split points and directions)
        """
        ret = dict()
        for atr in self.df_numeric.columns:
            if atr == "Fare":
                d = 0
            gains = [self._gini_gain(self.df_numeric[atr] > s, is_in_cluster) for s in self.numeric_split_sample[atr]]
            best_split_idx = np.argmax(next(zip(*gains)))
            best_gain, direction = gains[best_split_idx]
            best_bpoint = self.numeric_split_sample[atr][best_split_idx]
            best_split = [(best_bpoint, direction)]

            # TODO: refactor
            if self.find_intervals:
                gains = []

                for s in self.numeric_split_sample[atr]:
                    if direction and s > best_bpoint:
                        bool_atr = (self.df_numeric[atr] < s) & (self.df_numeric[atr] > best_bpoint)
                    elif not direction and s < best_bpoint:
                        bool_atr = (self.df_numeric[atr] < best_bpoint) & (self.df_numeric[atr] > s)
                    else:
                        continue

                    gains.append((self._gini_gain(bool_atr, is_in_cluster)[0], s))

                best_split_idx = np.argmax(next(zip(*gains)))
                best_interval_gain, best_bpoint2 = gains[best_split_idx]
                if best_interval_gain > best_gain:
                    best_split.append((best_bpoint2, not direction))
                    best_split.sort(key=lambda s: s[1], reverse=True)
                    best_gain = best_interval_gain

            ret[atr] = (best_gain, best_split)

        return ret

    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        discrete_gains = self._discrete_gains(is_in_cluster)
        continuous_gains = self._continuous_gains(is_in_cluster)
        gains = {**discrete_gains, **continuous_gains}

        # threshold and select top max_labels attributes
        total_impurity, _ = self._gini_impurity(is_in_cluster)
        filtered_attributes = [atr for atr, g in gains.items() if g[0] / total_impurity > self.g_threshold]
        selected_attributes = list(sorted(filtered_attributes, key=lambda k: gains[k][0], reverse=True))[:max_labels]

        labels = []
        for atr in selected_attributes:
            if atr in discrete_gains:
                values = list(map(str, discrete_gains[atr][1]))
                if len(values) > 1:
                    values[-1] = ' or ' + values[-1]

                label = '{} = {}'.format(atr, ', '.join(values[:-1]) + values[-1])

            else:
                split_points = continuous_gains[atr][1]
                if len(split_points) == 1:
                    break_point, dir = split_points[0]
                    label = '{} {} {:.2f}'.format(atr, '>' if dir else '<', break_point)
                else:
                    start, end = split_points[0][0], split_points[1][0]
                    label = '{:.2f} < {} <{:.2f}'.format(start, atr, end)

            labels.append(label)

        return labels
