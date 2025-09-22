import numpy as np

from typing import Callable
from sklearn.base import BaseEstimator
from fast_hbcc.hbcc import check_greater_equal


from .base import MultiMSTMixin
from .lib import multi_boruvka, DescentIndex, PrecomputedIndex, make_csr_graph
from .kmst import validate_parameters as validate_parameters_kmst


class KMSTDescent(BaseEstimator, MultiMSTMixin):
    """
    An SKLEARN-style estimator for computing approximate k-MSTs of a dataset.
    Adapts the boruvka algorithm to look for k candidate edges per point, of
    which the k best per connected component are retained (up to epsilon times
    the shortest distance).

    See MultiMSTMixin for inherited methods.

    Parameters
    ----------
    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is used
        it must be a numba njit compiled function. See the pynndescent docs for
        supported metrics. Metrics that take arguments (such as minkowski,
        mahalanobis etc.) can have arguments passed via the metric_kwds
        dictionary.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. At this time care must be taken and dictionary
        elements must be ordered appropriately; this will hopefully be fixed in
        the future.

    num_neighbors: int, optional
        The number of edges to connect between each fragment. Default is 3.

    min_samples: int, optional
        The number of neighbors to use for computing core distances. Default is
        1.

    epsilon: float, optional
        A fraction of the initial MST edge distance to act as upper distance
        bound.

    min_descent_neighbors: int, optional
        Runs the descent algorithm with more neighbors than we retain in the MST
        to improve convergence when num_neighbors is low. Default is 12.

    nn_kwargs: dict
        Additional keyword arguments to pass to NNDescent.

    Attributes
    ----------
    graph_ : scipy.sparse.csr_array
        The computed k-minimum spanning tree as sparse matrix with raw distance
        edge weights. Rows are sorted in ascending distance.

    mutual_reachability_graph_ : scipy.sparse.csr_array
        The computed k-minimum spanning tree as sparse matrix with mutual
        reachability edge weights. Rows are sorted in ascending distance.

    minimum_spanning_tree_ : numpy.ndarray, shape (n_points - 1, 3)
        A minimum spanning tree edgelist with raw distances (unsorted).

    mutual_reachability_tree_ : numpy.ndarray, shape (n_points - 1, 3)
        A minimum spanning tree edgelist with mutual reachability distances
        (unsorted).

    knn_neighbors_ : numpy.ndarray, shape (n_samples, num_neighbors)
        The kNN indices of the input data.

    knn_distances_ : numpy.ndarray, shape (n_samples, num_neighbors)
        The kNN (raw) distances of the input data.

    kmst_neighbors_: numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST edges in kNN format, -1 marks invalid indices.

    kmst_distances_: numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST (raw) distances in kNN format.
    """

    def __init__(
        self,
        metric: Callable | str = "euclidean",
        metric_kwds: dict | None = None,
        num_neighbors: int = 3,
        min_samples: int = 1,
        epsilon: float | None = None,
        min_descent_neighbors: int = 12,
        nn_kwargs: dict | None = None,
    ):
        super().__init__(metric=metric, metric_kwds=metric_kwds)
        self.num_neighbors = num_neighbors
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.min_descent_neighbors = min_descent_neighbors
        self.nn_kwargs = nn_kwargs

    def fit(self, X, y=None, **fit_params):
        """
        Computes the k-MST of the given data.

        Parameters
        ----------
        X: array-like
            The data to construct the MST for.

        y: array-like, optional
            Ignored.

        **fit_params: dict
            Ignored.

        Returns
        -------
        self: KMSTDescent
            The fitted estimator.
        """
        super().fit(X, y, **fit_params)
        clean_data = X if self._all_finite else X[self.finite_index]

        kwargs = self.get_params()
        (
            self._graph,
            self.minimum_spanning_tree_,
            self._knn_neighbors,
            self._knn_distances,
        ) = kMSTDescent(clean_data, **kwargs)

        super().remap_indices()
        return self


def kMSTDescent(
    data,
    metric: Callable | str = "euclidean",
    metric_kwds: dict | None = None,
    num_neighbors: int = 3,
    min_samples: int = 1,
    epsilon: float | None = None,
    min_descent_neighbors: int = 12,
    nn_kwargs: dict | None = None,
):
    """
    Computes an approximate k-MST of a dataset. Adapts the boruvka algorithm to
    look for k candidate edges per point, of which the k best per connected
    component are retained (up to epsilon times the shortest distance).

    Parameters
    ----------
    data: array-like
        Input data.

    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is used
        it must be a numba njit compiled function. See the pynndescent docs for
        supported metrics. Metrics that take arguments (such as minkowski,
        mahalanobis etc.) can have arguments passed via the metric_kwds
        dictionary. Precomputed distances can be passed to `data` as a 1D
        condensed or 2D square array, in which case the metric must be
        'precomputed'.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. At this time care must be taken and dictionary
        elements must be ordered appropriately; this will hopefully be fixed in
        the future.

    num_neighbors: int, optional
        The number of edges to connect between each fragment. Default is 3.

    min_samples: int, optional
        The number of neighbors to use for computing core distances. Default is
        1.

    epsilon: float, optional
        A fraction of the initial MST edge distance to act as upper distance
        bound.

    min_descent_neighbors: int, optional
        Runs the descent algorithm with more neighbors than we retain in the MST
        to improve convergence when num_neighbors is low. Default is 12.

    nn_kwargs: dict
        Additional keyword arguments to pass to NNDescent.

    Returns
    -------
    graph : scipy.sparse.csr_array
        The computed k-minimum spanning tree as sparse csr matrix with raw
        distance edge weights.

    minimum_spanning_tree : numpy.ndarray, shape (n_points - 1, 3)
        A minimum spanning tree edgelist with raw distances (unsorted).

    knn_indices : numpy.ndarray, shape (n_samples, num_neighbors)
        The kNN indices of the input data, -1 marks invalid indices.

    knn_distances : numpy.ndarray, shape (n_samples, num_neighbors)
        The kNN distances of the input data.
    """
    data, epsilon = validate_parameters(
        data, num_neighbors, min_samples, epsilon, min_descent_neighbors
    )
    if metric == "precomputed":
        index = PrecomputedIndex(
            data,
            num_neighbors,
            min_samples,
            min_descent_neighbors,
            nn_kwargs,
        )
    else:
        index = DescentIndex(
            data,
            metric,
            metric_kwds,
            num_neighbors,
            min_samples,
            min_descent_neighbors,
            nn_kwargs,
        )
    mst_edges, k_edges, neighbors, distances = multi_boruvka(index, epsilon)
    graph = make_csr_graph(mst_edges, k_edges, neighbors.shape[0])
    return (graph, mst_edges, neighbors, distances)


def validate_parameters(
    data, num_neighbors, min_samples, epsilon, min_descent_neighbors
):
    data, epsilon = validate_parameters_kmst(data, num_neighbors, min_samples, epsilon)
    check_greater_equal(
        min_value=1,
        datatype=np.integer,
        min_descent_neighbors=min_descent_neighbors,
    )
    return data, epsilon
