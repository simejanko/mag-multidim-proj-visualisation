import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from matplotlib import patches
from abc import ABC, abstractmethod


class BaseExplorer(ABC):
    """ Base class for visualisation tools for static and dynamic exploration of projections for different types of data. """

    STAT_FONT_SIZE_MAX = 18
    STAT_FONT_SIZE_MIN = 10
    DYNM_FONT_SIZE = 12

    def __init__(self, max_static_labels=3, max_dynamic_labels=5, min_cluster_size=5, fig_size=(12, 10)):
        """
        :param p_threshold: statistical significance (p-value) threshold for annotations
        :param max_static_labels: maximum number of static labels per cluster.
        :param max_dynamic_labels: maximum number of dynamic labels (shown for selected group of examples)
        :param min_cluster_size: minimum number of members in a cluster for showing a label
        :param fig_size: Figure size.
        """

        self.max_static_labels = max_static_labels
        self.max_dynamic_labels = max_dynamic_labels
        self.min_cluster_size = min_cluster_size

        self.fig_size = fig_size
        self.fig = None
        self.ax = None
        self.scatter_plot = None
        self.lens = None
        self.annotations = None

        self.X_em = None
        self.clusters = None
        self.kd_tree = None

        super().__init__()

    @abstractmethod
    def fit(self, data, X_em, clusters):
        """
        Performs any kind of preprocessing and caching needed for lens exploration.
        :param data: input data. Different type across subclasses.
        :param X_em: numpy array of embeddings with shape (n_samples, 2)
        :param clusters: numpy array of cluster labels with shape (n_samples,)
        """
        self.annotations = []

        self.X_em = X_em
        self.clusters = clusters

        self.kd_tree = KDTree(self.X_em, leaf_size=20)

    def _init_plot(self, ax = None):
        """ Initializes plot. """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.fig_size)
        else:
            self.ax = ax

        self.scatter_plot = None
        self.lens = plt.Circle((0, 0), 0, edgecolor='black', fill=False)
        self.ax.add_artist(self.lens)

    @abstractmethod
    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        pass

    def plot_static(self, classes=None, annotation_bg_alpha=0.5, plot_labels=True, ax=None, **kwargs):
        """
        Plots static labels for clusters of the embedding.
        :param classes: color array of shape (n_samples, ) that allows custom coloring of scatter plot based on class attribute.
        :param annotation_bg_alpha: transparency of annotation backgrounds.
        :param plot_labels: true if labels should be included in the visualisation, false otherwise.
        :param ax: specify existing matplotlib axis to use for this visualisation
        :param kwargs: optional other parameters to pass to matplotlib's scatterplot
        """
        self._init_plot(ax=ax)

        if classes is None:
            colors = [(0.5, 0.5, 0.5, 1)] * self.X_em.shape[0]
            legend_patches = []
        else:
            cmap = plt.get_cmap('tab10')
            le = LabelEncoder()
            y = le.fit_transform(classes)
            colors = cmap(y)
            legend_patches = [patches.Patch(color=cmap(le.transform([c])[0]), label=c) for c in le.classes_]

        self.scatter_plot = self.ax.scatter(self.X_em[:, 0], self.X_em[:, 1], facecolor=colors,
                                            edgecolor=[(0, 0, 0, 1)] * self.X_em.shape[0], **kwargs)
        self.ax.axis('equal')

        if plot_labels:
            for c in np.unique(self.clusters):
                if c < 0:
                    continue

                is_in_cluster = self.clusters == c

                if np.sum(is_in_cluster) < self.min_cluster_size:
                    continue

                labels = self._extract_labels(is_in_cluster, self.max_static_labels)

                x_avg, y_avg = np.mean(self.X_em[is_in_cluster, :], axis=0)
                font_sizes = np.linspace(self.STAT_FONT_SIZE_MAX, self.STAT_FONT_SIZE_MIN, num=self.max_static_labels)
                dy = [0, font_sizes[0] * 0.6]
                for i, label in enumerate(labels):
                    annotation_side = (i % 2 * 2 - 1)
                    if i>0:
                        dy[i % 2] += annotation_side * font_sizes[i] * 0.6

                    self.ax.annotate(label, (x_avg, y_avg), (0, dy[i % 2]), textcoords='offset points', ha='center',
                                     va='center', fontsize=font_sizes[i],
                                     bbox=dict(boxstyle='square,pad=0.1', facecolor='red', alpha=annotation_bg_alpha,
                                               linewidth=0),
                                     color='white', fontweight='bold')

                    dy[i % 2] += annotation_side * font_sizes[i] * 0.6

                #self.ax.text(x_avg, y_avg, '\n'.join(labels), ha='center', va='center', fontsize=np.mean(font_sizes)+1)

        if len(legend_patches) > 0:
            self.ax.legend(handles=legend_patches)

        return self.fig, self.ax

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

        is_selected = np.zeros(self.X_em.shape[0], bool)
        selected_idx = self.kd_tree.query_radius([[x, y]], r=r)[0]
        is_selected[selected_idx] = True
        if np.sum(is_selected) == 0:
            return self.fig, self.ax

        labels = self._extract_labels(is_selected, self.max_dynamic_labels)

        face_colors = self.scatter_plot.get_facecolor()
        edge_colors = self.scatter_plot.get_edgecolor()
        face_colors[is_selected, -1] = 1
        face_colors[~is_selected, -1] = 0.5
        edge_colors[is_selected, -1] = 1
        edge_colors[~is_selected, -1] = 0.5
        self.scatter_plot.set_facecolor(face_colors)
        self.scatter_plot.set_edgecolor(edge_colors)

        dy = 1.2 * self.DYNM_FONT_SIZE * len(labels) / 2
        for i, label in enumerate(labels):
            ann = self.ax.annotate(label, (x - r, y), (-3, dy), textcoords='offset points', ha='right',
                                   va='center', fontsize=self.DYNM_FONT_SIZE,
                                   bbox=dict(boxstyle='square,pad=0.1', fill=False),
                                   color='black')
            self.annotations.append(ann)
            dy -= 1.2 * self.DYNM_FONT_SIZE
        return self.fig, self.ax

    def plot_interactive(self, r=5):
        """
        Connect click event with dynamic label updating for interactivity.
        :param r: lens radius
        """

        def onclick(event):
            if event.button != 1:
                return

            self.plot_dynamic(event.xdata, event.ydata, self.lens.get_radius())

        self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.lens.set_radius(r)
        return self.fig, self.ax
