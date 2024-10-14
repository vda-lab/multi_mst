import numpy as np
import warnings as warn
from umap import UMAP
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from sklearn.base import BaseEstimator

from .boruvka import parallel_boruvka
from ..kdtree import kdtree_to_numba


def validate_parameters(data, num_neighbors, min_samples, epsilon):
    data = check_array(data)
    if (
        not (np.issubdtype(type(num_neighbors), np.integer) or num_neighbors is None)
    ) or (num_neighbors is not None and num_neighbors < 1):
        raise ValueError("num_neighbors must be an integer >= 1.")

    if (not (np.issubdtype(type(min_samples), np.integer) or min_samples is None)) or (
        min_samples is not None and (min_samples < 1 or min_samples > num_neighbors)
    ):
        raise ValueError("min_samples must be an integer >= 1 <= num_neighbors.")

    if epsilon is None:
        epsilon = np.inf
    elif not np.issubdtype(type(epsilon), np.floating) or epsilon < 1.0:
        raise ValueError("epsilon must be None or a floating point number >= 1.0")

    return data, epsilon


def kMST(data, num_neighbors=3, min_samples=1, epsilon=None, umap_kwargs=None):
    """
    Computes a k-MST of a dataset. Adapts the boruvka algorithm to look for
    k candidate edges per point, of which the k best per connected component
    are retained (up to epsilon times the shortest distance).

    The algorithm operates on HDBSCAN's mutual reachability Euclidean distance.
    The resulting graph is embedded with UMAP as if it contains normal k nearest
    neighbors.

    Parameters
    ----------
    data: array-like
        The data to construct a MST for.
    num_neighbors: int, optional
        The number of edges to connect between each fragement. Default is 3.
    min_samples: int, optional
        The number of neighbors for computing the mutual reachability distance.
        Value must be lower or equal to the number of neighbors. `epsilon`
        operates on the mutual reachability distance, so always allows the
        nearest `min_samples` points. Acts as UMAP's `local connnectivity`
        parameter. Default is 1.
    epsilon: float, optional
        A fraction of the initial MST edge distance to act as upper distance
        bound.
    umap_kwargs: dict
        Additional keyword arguments to pass to UMAP.

    Returns
    -------
    mst_indices_: numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST edges in kNN format.
    mst_distances_: numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST edges in kNN format.
    umap: umap.UMAP
        A fitted UMAP object.
    """
    if umap_kwargs is None:
        umap_kwargs = {}
    (data, epsilon) = validate_parameters(data, num_neighbors, min_samples, epsilon)
    sklearn_tree = KDTree(data)
    numba_tree = kdtree_to_numba(sklearn_tree)
    mst_indices, mst_distances = parallel_boruvka(
        numba_tree, num_neighbors, min_samples, epsilon
    )
    with warn.catch_warnings():
        warn.filterwarnings(
            "ignore", category=UserWarning, module="umap.umap_", lineno=2010
        )
        umap = UMAP(
            n_neighbors=mst_indices.shape[1],
            precomputed_knn=(mst_indices, mst_distances),
            **umap_kwargs
        ).fit(data)
    return (mst_indices, mst_distances, umap)


class KMST(BaseEstimator):
    """
    An SKLEARN-style estimator for computing a k-MST of a dataset. Adapts the
    boruvka algorithm to look for k candidate edges per point, of which the
    k best per connected component are retained (up to epsilon times the
    shortest distance).

    The algorithm operates on HDBSCAN's mutual reachability Euclidean distance.
    The resulting graph is embedded with UMAP as if it contains normal k nearest
    neighbors.

    Parameters
    ----------
    num_neighbors: int, optional
        The number of edges to connect between each fragement. Default is 35.
    min_samples: int, optional
        The number of neighbors for computing the mutual reachability distance.
        Value must be lower or equal to the number of neighbors. `epsilon`
        operates on the mutual reachability distance, so always allows the
        nearest `min_samples` points. Acts as UMAP's `local connnectivity`
        parameter. Default is 1.
    epsilon: float, optional
        A fraction of the initial MST edge distance to act as upper distance
        bound.
    umap_kwargs: dict
        Additional keyword arguments to pass to UMAP.

    Attributes
    ----------
    graph_: scipy.sparse matrix
        The computed k-minimum spanning tree as sparse matrix with edge weights
        as UMAP edge probability.
    embedding_: numpy.ndarray, shape (n_samples, num_components)
        The low dimensional embedding of the data. None if transform_mode is
        'graph'.
    mst_indices_: numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST edges in kNN format.
    mst_distances_: numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST edges in kNN format.
    _umap: umap.UMAP
        A fitted UMAP object used to compute the graph and embedding.
        Observations with infinite or missing values are not passed to the UMAP
        algorithm. The object's attributes are not adjusted to account for these
        missing values. Use the graph_ and embedding_ attributes instead!
    """

    def __init__(self, *, num_neighbors=3, min_samples=1, epsilon=None, umap_kwargs=None):
        self.num_neighbors = num_neighbors
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.umap_kwargs = umap_kwargs

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
        X = check_array(X, accept_sparse="csr", force_all_finite=False)
        self._raw_data = X

        self._all_finite = np.all(np.isfinite(X))
        if ~self._all_finite:
            # Pass only the purely finite indices into the algorithm, we will
            # restore indices later
            finite_index = np.where(np.isfinite(X).sum(axis=1) == X.shape[1])[0]
            clean_data = X[finite_index]
            internal_to_raw = {
                x: y for x, y in zip(range(len(finite_index)), finite_index)
            }
        else:
            clean_data = X

        kwargs = self.get_params()
        self.mst_indices_, self.mst_distances_, self._umap = kMST(
            clean_data, **kwargs
        )
        self.graph_ = self._umap.graph_.copy()
        self.embedding_ = (
            self._umap.embedding_.copy() if hasattr(self._umap, "embedding_") else None
        )

        if not self._all_finite:
            self.graph_ = self.graph_.tocoo()
            for i in range(len(self.graph_.data)):
                self.graph_.row[i] = internal_to_raw[self.graph_.row[i]]
                self.graph_.col[i] = internal_to_raw[self.graph_.col[i]]
            self.graph_ = coo_matrix(
                (self.graph_.data, (self.graph_.row, self.graph_.col)),
                shape=(X.shape[0], X.shape[0]),
            )

            if self.embedding_ is not None:
                new_embedding = np.full((X.shape[0], self.embedding_.shape[1]), np.nan)
                new_embedding[finite_index] = self.embedding_
                self.embedding_ = new_embedding

            new_indices = np.full(
                (X.shape[0], self.mst_indices_.shape[1]), -1, dtype=np.int32
            )
            new_indices[finite_index] = self.mst_indices_
            self.mst_indices_ = new_indices

            new_distances = np.full(
                (X.shape[0], self.mst_distances_.shape[1]), np.inf, dtype=np.float32
            )
            new_distances[finite_index] = self.mst_distances_
            self.mst_distances_ = new_distances

        return self

    def fit_transform(self, X, y=None, **fit_params):
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
        embedding: numpy.ndarray, shape (n_samples, num_components)
            The computed low dimensional embedding. None if transform_mode is
            'graph'.
        """
        self.fit(X, y=y, **fit_params)
        return self.embedding_
