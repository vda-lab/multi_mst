import numba
import numpy as np
from pynndescent import NNDescent
from fast_hdbscan.sub_clusters import SubClusterDetector
from fast_hdbscan.branches import apply_branch_threshold


def make_compute_centrality(metric, metric_kwds):
    # Use NNDescent to find the same metric as KMSTDescent
    index = NNDescent.__new__(NNDescent, metric=metric, metric_kwds=metric_kwds)
    index.metric = metric
    index._dist_args = tuple(dict().values())
    index._set_distance_func()
    distance_fun = index._distance_func
    correction = index._distance_correction

    # Define the distance function
    @numba.njit(parallel=True)
    def dist_fun(X, centroid):
        result = np.empty(X.shape[0], dtype=np.float32)
        for i in numba.prange(X.shape[0]):
            result[i] = distance_fun(X[i], centroid)
        if correction is not None:
            return correction(result)
        return result

    # Compute centrality with the distance function
    def compute_centrality(data, probabilities, *args):
        points = args[-1]
        cluster_data = data[points, :].astype(np.float32)
        centroid = np.average(
            cluster_data, weights=probabilities[points], axis=0
        ).astype(np.float32)
        with np.errstate(divide="ignore"):
            return 1 / dist_fun(cluster_data, centroid)

    return compute_centrality


class BranchDetector(SubClusterDetector):
    def __init__(
        self,
        *,
        metric="euclidean",
        metric_kwds=None,
        label_sides_as_branches=False,
        min_cluster_size=None,
        max_cluster_size=None,
        allow_single_cluster=None,
        cluster_selection_method=None,
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
        propagate_labels=False,
    ):
        super().__init__(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            propagate_labels=propagate_labels,
        )
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.label_sides_as_branches = label_sides_as_branches

    def fit(self, clusterer, labels=None, probabilities=None, sample_weight=None):
        super().fit(
            clusterer,
            labels,
            probabilities,
            sample_weight,
            make_compute_centrality(self.metric, self.metric_kwds),
        )
        apply_branch_threshold(
            self.labels_,
            self.sub_cluster_labels_,
            self.probabilities_,
            self.cluster_probabilities_,
            self.cluster_points_,
            self._linkage_trees,
            label_sides_as_branches=self.label_sides_as_branches,
        )
        self.branch_labels_ = self.sub_cluster_labels_
        self.branch_probabilities_ = self.sub_cluster_probabilities_
        self.centralities_ = self.lens_values_
        return self

    def propagated_labels(self, label_sides_as_branches=None):
        if label_sides_as_branches is None:
            label_sides_as_branches = self.label_sides_as_branches

        labels, branch_labels = super().propagated_labels()
        apply_branch_threshold(
            labels,
            branch_labels,
            np.zeros_like(self.probabilities_),
            np.zeros_like(self.probabilities_),
            self.cluster_points_,
            self._linkage_trees,
            label_sides_as_branches=label_sides_as_branches,
        )
        return labels, branch_labels

    @property
    def approximation_graph_(self):
        """See :class:`~hdbscan.plots.ApproximationGraph` for documentation."""
        return super()._make_approximation_graph(
            lens_name="centrality", sub_cluster_name="branch"
        )
