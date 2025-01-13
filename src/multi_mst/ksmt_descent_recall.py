from sklearn.base import BaseEstimator

from .base import MultiMSTMixin
from .lib import multi_boruvka_recall, KDTreeIndex, DescentRecallIndex, make_csr_graph
from .kmst_descent import validate_parameters


class KMSTDescentLogRecall(BaseEstimator, MultiMSTMixin):
    """
    An SKLEARN-style estimator for computing approximate k-MSTs. This version
    keeps track of the recall between ground-truth and the approximate MST
    edges. Recall is computed in every descent iteration and in every Boruvka
    iteration.

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

    min_descent_neighbors: int, optional
        Runs the descent algorithm with more neighbors than we retain in the MST
        to improve convergence when num_neighbors is low. Default is 12.

    nn_kwargs: dict
        Additional keyword arguments to pass to NNDescent.

    Attributes
    ----------
    trace_ : list[dict]
        A trace with metrics of the nn-descent algorithm.

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
        num_neighbors: int = 3,
        min_samples: int = 1,
        epsilon: float | None = None,
        min_descent_neighbors: int = 12,
        nn_kwargs: dict | None = None,
    ):
        super().__init__()
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
            self.trace_,
        ) = kMSTDescentLogRecall(clean_data, **kwargs)

        super().remap_indices()
        return self


def kMSTDescentLogRecall(
    data,
    num_neighbors=3,
    min_samples=1,
    epsilon=None,
    min_descent_neighbors=12,
    nn_kwargs=None,
):
    """
    Computes an approximate k-MST of a dataset. Adapts the boruvka algorithm to
    look for k candidate edges per point, of which the k best per connected
    component are retained (up to epsilon times the shortest distance).

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

    min_descent_neighbors: int, optional
        Runs the descent algorithm with more neighbors than we retain in the MST
        to improve convergence when num_neighbors is low. Default is 6.

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

    trace: list[dict]
        A trace of the recall across Boruvka iterations (outer) and nn descent
        iterations (inner).
    """
    data, epsilon = validate_parameters(
        data, num_neighbors, min_samples, epsilon, min_descent_neighbors
    )
    true_index = KDTreeIndex(data, num_neighbors, min_samples)
    recall_index = DescentRecallIndex(
        data,
        num_neighbors=num_neighbors,
        min_samples=min_samples,
        min_descent_neighbors=min_descent_neighbors,
        nn_kwargs=nn_kwargs,
    )
    mst_edges, k_edges, neighbors, distances, trace = multi_boruvka_recall(
        true_index, recall_index, epsilon
    )
    trace = [
        {
            "boruvka_num_components": t[0],
            "boruvka_recall": t[1],
            "descent_recall": t[2],
            "descent_distance_fraction": t[3],
            "descent_num_changes": t[4],
        }
        for t in trace
    ]
    graph = make_csr_graph(mst_edges, k_edges, neighbors.shape[0])
    return (graph, mst_edges, neighbors, distances, trace)
