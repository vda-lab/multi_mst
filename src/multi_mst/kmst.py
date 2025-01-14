import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from fast_hbcc.hbcc import check_greater_equal

from .base import MultiMSTMixin
from .lib import multi_boruvka, KDTreeIndex, make_csr_graph


class KMST(BaseEstimator, MultiMSTMixin):
    """
    An SKLEARN-style estimator for computing a k-MST of a dataset. Adapts the
    boruvka algorithm to look for k candidate edges per point, of which the
    k best per connected component are retained (up to epsilon times the
    shortest distance).

    See MultiMSTMixin for inherited methods.

    Parameters
    ----------
    num_neighbors: int, optional
        The number of edges to connect between each fragment. Default is 3.

    min_samples: int, optional
        The number of neighbors to use for computing core distances. Default is
        1.

    epsilon: float, optional
        A fraction of the initial MST edge distance to act as upper distance
        bound.

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
        self, num_neighbors: int = 3, min_samples: int = 1, epsilon: float | None = None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.min_samples = min_samples
        self.epsilon = epsilon

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
        self: KMST
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
        ) = kMST(clean_data, **kwargs)

        super().remap_indices()
        return self


def kMST(
    data, num_neighbors: int = 3, min_samples: int = 1, epsilon: float | None = None
):
    """
    Computes a k-MST of a dataset. Adapts the boruvka algorithm to look for
    k candidate edges per point, of which the k best per connected component
    are retained (up to epsilon times the shortest distance).

    Parameters
    ----------
    data: array-like
        Input data.

    num_neighbors: int, optional
        The number of edges to connect between each fragment. Default is 3.

    min_samples: int, optional
        The number of neighbors to use for computing core distances. Default is
        1.

    epsilon: float, optional
        A fraction of the initial MST edge distance to act as upper distance
        bound.

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
    data, epsilon = validate_parameters(data, num_neighbors, min_samples, epsilon)
    index = KDTreeIndex(data, num_neighbors, min_samples)
    mst_edges, k_edges, neighbors, distances = multi_boruvka(index, epsilon)
    graph = make_csr_graph(mst_edges, k_edges, neighbors.shape[0])
    return (graph, mst_edges, neighbors, distances)


def validate_parameters(data, num_neighbors, min_samples, epsilon):
    data = check_array(data)
    if epsilon is None:
        epsilon = np.inf
    check_greater_equal(
        min_value=1,
        datatype=np.integer,
        num_neighbors=num_neighbors,
        min_samples=min_samples,
    )
    check_greater_equal(min_value=1.0, datatype=np.floating, epsilon=epsilon)
    if min_samples > num_neighbors:
        raise ValueError("min_samples must be less than or equal to num_neighbors.")
    return data, epsilon
