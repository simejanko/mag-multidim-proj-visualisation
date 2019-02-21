import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from matplotlib import patches
from abc import ABC, abstractmethod
from adjustText import adjust_text
from scipy.spatial.distance import euclidean
from lib.utils.geometry import rectangle_circle_bbox_intersect, rectangle_intersect
from matplotlib.colors import ListedColormap
from itertools import chain

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
        self.static_annotations = None
        self.dynamic_annotations = None

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
        self.static_annotations = set()
        self.dynamic_annotations = set()

        self.X_em = X_em
        self.clusters = clusters

        self.kd_tree = KDTree(self.X_em, leaf_size=20)

    def _init_plot(self, ax=None):
        """ Initializes plot. """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.fig_size)
        else:
            self.ax = ax

        self.scatter_plot = None
        self.lens = plt.Circle((0, 0), 0, edgecolor='black', fill=False)
        self.ax.add_artist(self.lens)

        self.static_annotations = set()

    @abstractmethod
    def _extract_labels(self, is_in_cluster, max_labels):
        """
        Get labels for a given cluster.
        :param is_in_cluster: boolean array of shape (n_samples, ) that indicates cluster membership
        :param max_labels: maximum number of labels to extract
        :return: list of labels
        """
        pass

    def _obtain_static_labels_and_bounding_boxes(self):
        """
        Obtains static labels for all clusters as well as corresponding matplotlib Text objects
        with approximate bounding boxes later used by label overlap resolver
        :return list of tuples (label_group, (x,y)) and list of corresponding matplotlib Text objects
        """
        text_objects = []
        all_labels = []

        for c in np.unique(self.clusters):
            if c < 0:
                continue

            is_in_cluster = self.clusters == c

            if np.sum(is_in_cluster) < self.min_cluster_size:
                continue

            labels = self._extract_labels(is_in_cluster, self.max_static_labels)
            x_avg, y_avg = np.mean(self.X_em[is_in_cluster, :], axis=0)
            all_labels.append((labels, (x_avg, y_avg)))

            #
            text = self.ax.text(x_avg, y_avg, '\n'.join(labels), ha='center', va='center',
                                fontsize=1 + (self.STAT_FONT_SIZE_MAX + self.STAT_FONT_SIZE_MIN) / 2, alpha=0)
            text_objects.append(text)

        return text_objects, all_labels

    def _plot_static_labels(self, annotation_bg_alpha, avoid_overlaps, displacement_lines):
        """
        Plots static labels for clusters of the embedding.
        :param annotation_bg_alpha: transparency of annotation backgrounds.
        :param avoid_overlaps: tries to avoid static label overlaps but may lead to inaccurate label placements
        :param displacement_lines: adds soft lines to indicate displacements made by avoiding overlaps.
        """

        text_objects, all_labels = self._obtain_static_labels_and_bounding_boxes()

        # displace labels to resolve overlaps
        if avoid_overlaps:
            adjust_text(text_objects, ax=self.ax, text_from_points=False, autoalign=False, force_text=0.1)

        # draw actual annotations on displaced locations
        for (labels, (x_orig, y_orig)), text in zip(all_labels, text_objects):
            x, y = text.get_position()
            text.remove()

            font_sizes = np.linspace(self.STAT_FONT_SIZE_MAX, self.STAT_FONT_SIZE_MIN, num=self.max_static_labels)
            dy = [0, font_sizes[0] * 0.6]
            for i, label in enumerate(labels):
                annotation_side = (i % 2 * 2 - 1)
                if i > 0:
                    dy[i % 2] += annotation_side * font_sizes[i] * 0.6

                if displacement_lines:
                    self.ax.arrow(x, y, x_orig - x, y_orig - y, edgecolor=(1.0, 0.0, 0.0, annotation_bg_alpha / 2),
                                  width=1e-6)

                ann = self.ax.annotate(label, (x, y), (0, dy[i % 2]), textcoords='offset points',
                                       ha='center',
                                       va='center', fontsize=font_sizes[i],
                                       bbox=dict(boxstyle='square,pad=0.1', facecolor='red',
                                                 alpha=annotation_bg_alpha,
                                                 linewidth=0),
                                       color='white', fontweight='bold')
                self.static_annotations.add(ann)

                dy[i % 2] += annotation_side * font_sizes[i] * 0.6

    def plot_static(self, classes=None, annotation_bg_alpha=0.5, plot_labels=True, avoid_overlaps=False,
                    displacement_lines=True, ax=None, **kwargs):
        """
        Plots scatter plot and static labels for clusters of the embedding.
        :param classes: color array of shape (n_samples, ) that allows custom coloring of scatter plot based on class attribute.
        :param annotation_bg_alpha: transparency of annotation backgrounds.
        :param plot_labels: true if labels should be included in the visualisation, false otherwise.
        :param avoid_overlaps: tries to avoid static label overlaps but may lead to inaccurate label placements
        :param displacement_lines: adds soft lines to indicate displacements made by avoiding overlaps.
        :param ax: specify existing matplotlib axis to use for this visualisation
        :param kwargs: optional other parameters to pass to matplotlib's scatterplot
        """
        self._init_plot(ax=ax)

        if classes is None:
            colors = [(0.5, 0.5, 0.5, 1)] * self.X_em.shape[0]
            legend_patches = []
        else:
            #merge multiple discrete cmaps
            cmaps = [plt.get_cmap('tab10'), plt.get_cmap('Set1'), plt.get_cmap('Set2'), plt.get_cmap('Set3'),
                     plt.get_cmap('Accent'), plt.get_cmap('Dark2')]
            cmap = ListedColormap(list(chain.from_iterable(cm.colors for cm in cmaps)))

            le = LabelEncoder()
            y = le.fit_transform(classes)
            colors = cmap(y)

            legend_patches = [patches.Patch(color=cmap(le.transform([c])[0]), label=c) for c in le.classes_]

        self.scatter_plot = self.ax.scatter(self.X_em[:, 0], self.X_em[:, 1], facecolor=colors,
                                            edgecolor=[(0, 0, 0, 1)] * self.X_em.shape[0], **kwargs)
        self.ax.axis('equal')

        if plot_labels:
            self._plot_static_labels(annotation_bg_alpha, avoid_overlaps, displacement_lines)

        if len(legend_patches) > 0:
            self.ax.legend(handles=legend_patches)

        return self.fig, self.ax

    def _resolve_dynamic_overlaps(self):
        """
        Hides static labels that intersect with lens.
        """
        for sa in self.static_annotations:
            sa.set_visible(True)
            sa_bbox = sa.get_window_extent()
            sa_bbox_t = sa.get_window_extent().transformed(self.ax.transData.inverted())

            sa_bbox = sa_bbox.x0, sa_bbox.y0, sa_bbox.width, sa_bbox.height
            sa_bbox_t = sa_bbox_t.x0, sa_bbox_t.y0, sa_bbox_t.width, sa_bbox_t.height

            lens_circle = self.lens.center + (self.lens.get_radius(), )

            if rectangle_circle_bbox_intersect(sa_bbox_t, lens_circle):
                sa.set_visible(False)
                continue

            #TODO: this can probably be done way more efficiently (eg. kd-tree) but we don't expect huge number of labels
            for da in self.dynamic_annotations:
                da_bbox = da.get_window_extent()
                da_bbox = da_bbox.x0, da_bbox.y0, da_bbox.width, da_bbox.height

                if rectangle_intersect(sa_bbox, da_bbox):
                    sa.set_visible(False)
                    break


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
        for ann in self.dynamic_annotations:
            ann.remove()
        self.dynamic_annotations = set()

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
                                   bbox=dict(boxstyle='square,pad=0.1', fill=False,
                                             edgecolor='black'),
                                   color='black')
            self.dynamic_annotations.add(ann)
            dy -= 1.2 * self.DYNM_FONT_SIZE

        self._resolve_dynamic_overlaps()
        return self.fig, self.ax

    def plot_interactive(self, r=5):
        """
        Connect click event with dynamic label updating for interactivity.
        :param r: lens radius
        """

        def onclick(event):
            # left click - move lens
            if event.button == 1:
                self.plot_dynamic(event.xdata, event.ydata, self.lens.get_radius())
            # right click - change radius
            elif event.button == 3:
                x, y = self.lens.center
                new_r = euclidean((x, y), (event.xdata, event.ydata))
                self.plot_dynamic(x, y, new_r)

        self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.lens.set_radius(r)
        return self.fig, self.ax
