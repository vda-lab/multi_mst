import math
import numba
import numpy as np
import warnings as warn
from typing import Literal, Any, Callable

from time import time
from numpy.random import RandomState
from scipy.sparse import csr_array, spmatrix
from sklearn.utils import check_array
from sklearn.decomposition import PCA
from sklearn.manifold._t_sne import TSNE, _joint_probabilities_nn
from sklearn.utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    check_random_state,
)

from umap import UMAP
from fast_hbcc.sub_clusters import BoundaryClusterDetector
from fast_hbcc.hbcc import (
    HBCC,
    remap_csr_graph,
    remap_core_graph,
    compute_boundary_coefficient,
    check_greater_equal,
    check_optional_size,
    check_literals,
)
from fast_hdbscan.hdbscan import (
    HDBSCAN,
    remap_condensed_tree,
    remap_single_linkage_tree,
    to_numpy_rec_array,
)

from .lib import (
    BranchDetector,
    csr_to_neighbor_list,
    csr_mutual_reachability,
    edgelist_mutual_reachability,
    clusters_from_spanning_tree,
    lensed_spanning_tree_from_graph,
)


class MultiMSTMixin:
    """
    A base class implementing shared functionality for multi spanning tree
    classes.

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
    graph_neighbors_ : numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST edges in kNN format, -1 marks invalid indices.
    graph_distances_ : numpy.ndarray, shape (n_samples, num_found_neighbors)
        The kMST (raw) distances in kNN format.
    """

    def __init__(self, metric="euclidean", metric_kwds=None):
        self.metric = metric
        self.metric_kwds = metric_kwds

    def fit(self, X, y=None, **fit_params):
        """
        Manages the infinite data handling.

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
        self: MultiMSTMixin
            The fitted estimator.
        """
        X = check_array(X, ensure_all_finite=False)
        self._raw_data = X

        self._all_finite = np.all(np.isfinite(X))
        if ~self._all_finite:
            self.finite_index = np.where(np.isfinite(X).sum(axis=1) == X.shape[1])[0]
            self.internal_to_raw = {
                x: y for x, y in zip(range(len(self.finite_index)), self.finite_index)
            }

        return self

    def remap_indices(self):
        """Remaps the indices of the kNN and kMST graphs to the original raw
        data."""
        if self._all_finite:
            self.graph_ = self._graph
            self.knn_neighbors_ = self._knn_neighbors
            self.knn_distances_ = self._knn_distances
        else:
            new_indices, new_indptr = remap_csr_graph(
                self._graph,
                self.finite_index,
                self.internal_to_raw,
                self._raw_data.shape[0],
            )
            self.graph_ = csr_array(
                (self._graph.data, new_indices, new_indptr),
                shape=(self._raw_data.shape[0], self._raw_data.shape[0]),
            )

            new_indices = np.full(
                (self._raw_data.shape[0], self._knn_neighbors.shape[1]),
                -1,
                dtype=np.int32,
            )
            for row, neighbors in enumerate(self._knn_neighbors):
                for i, neighbor in enumerate(neighbors):
                    if neighbor == -1:
                        break
                    new_indices[row, i] = self.internal_to_raw[neighbor]
            self.knn_neighbors_ = new_indices

            new_distances = np.full(
                (self._raw_data.shape[0], self._knn_distances.shape[1]),
                np.inf,
                dtype=np.float32,
            )
            new_distances[self.finite_index] = self._knn_distances
            self.knn_distances_ = new_distances

        self._dirty = True

    @property
    def mutual_reachability_graph_(self):
        check_is_fitted(
            self,
            ["graph_", "_knn_distances"],
            msg="You first need to fit the estimator before accessing attributes.",
        )
        return csr_mutual_reachability(self.graph_.copy(), self._knn_distances.T[-1])

    @property
    def mutual_reachability_tree_(self):
        check_is_fitted(
            self,
            ["minimum_spanning_tree_", "_knn_distances"],
            msg="You first need to fit the estimator before accessing attributes.",
        )
        return edgelist_mutual_reachability(
            self.minimum_spanning_tree_.copy(), self._knn_distances.T[-1]
        )

    @property
    def graph_neighbors_(self):
        self._make_graph_neighbors()
        return self._graph_indices

    @property
    def graph_distances_(self):
        self._make_graph_neighbors()
        return self._graph_distances

    def _make_graph_neighbors(self):
        check_is_fitted(
            self,
            ["_graph", "_dirty"],
            msg="You first need to fit the estimator before accessing attributes.",
        )
        if self._dirty:
            self._graph_indices, self._graph_distances = csr_to_neighbor_list(
                self.graph_.data, self.graph_.indices, self.graph_.indptr
            )
            self._dirty = False

    def umap(
        self,
        *,
        n_components: int = 2,
        output_metric: Callable | str = "euclidean",
        output_metric_kwds: dict | None = None,
        n_epochs: int | None = None,
        learning_rate: float = 1.0,
        init: str | Any = "spectral",
        min_dist: float = 0.1,
        spread: float = 1.0,
        set_op_mix_ratio: float = 1.0,
        local_connectivity: float = 1.0,
        repulsion_strength: float = 1.0,
        negative_sample_rate: int = 5,
        a: float | None = None,
        b: float | None = None,
        random_state: int | Any | None = None,
        target_n_neighbors: int = -1,
        target_metric: Callable | str = "categorical",
        target_metric_kwds: dict | None = None,
        target_weight: float = 0.5,
        transform_seed: int = 42,
        transform_mode: Literal["embedding", "graph"] = "embedding",
        verbose: bool = False,
        tqdm_kwds: dict | None = None,
        densmap: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        output_dens: bool = False,
        disconnection_distance: float | None = None,
    ):
        """Constructs and fits a UMAP model [1]_ to the kMST graph.

        Unlike HDBSCAN and HBCC, UMAP does not support infinite data. To ensure
        all UMAP's member functions work as expected, the UMAP model is NOT
        remapped to the infinite data after fitting. As a result, combining UMAP
        and HDBSCAN results need to consider the finite index:
        ```
            plt.scatter(*umap.embedding_.T, c=hdbscan.labels_[multi_mst.finite_index])
        ```

        Parameters
        ----------
        n_components: int (optional, default 2)
            The dimension of the space to embed into. This defaults to 2 to
            provide easy visualization, but can reasonably be set to any integer
            value in the range 2 to 100.

        n_epochs: int (optional, default None)
            The number of training epochs to be used in optimizing the low
            dimensional embedding. Larger values result in more accurate
            embeddings. If None is specified a value will be selected based on
            the size of the input dataset (200 for large datasets, 500 for
            small).

        learning_rate: float (optional, default 1.0)
            The initial learning rate for the embedding optimization.

        init: string (optional, default 'spectral')
            How to initialize the low dimensional embedding. Options are:

                * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
                * 'random': assign initial embedding positions at random.
                * 'pca': use the first n_components from PCA applied to the
                    input data.
                * 'tswspectral': use a spectral embedding of the fuzzy
                    1-skeleton, using a truncated singular value decomposition
                    to "warm" up the eigensolver. This is intended as an
                    alternative to the 'spectral' method, if that takes an
                    excessively long time to complete initialization (or fails
                    to complete).
                * A numpy array of initial embedding positions.

        min_dist: float (optional, default 0.1)
            The effective minimum distance between embedded points. Smaller
            values will result in a more clustered/clumped embedding where
            nearby points on the manifold are drawn closer together, while
            larger values will result on a more even dispersal of points. The
            value should be set relative to the ``spread`` value, which
            determines the scale at which embedded points will be spread out.

        spread: float (optional, default 1.0)
            The effective scale of embedded points. In combination with
            ``min_dist`` this determines how clustered/clumped the embedded
            points are.

        set_op_mix_ratio: float (optional, default 1.0)
            Interpolate between (fuzzy) union and intersection as the set
            operation used to combine local fuzzy simplicial sets to obtain a
            global fuzzy simplicial sets. Both fuzzy set operations use the
            product t-norm. The value of this parameter should be between 0.0
            and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will
            use a pure fuzzy intersection.

        local_connectivity: int (optional, default 1)
            The local connectivity required -- i.e. the number of nearest
            neighbors that should be assumed to be connected at a local level.
            The higher this value the more connected the manifold becomes
            locally. In practice this should be not more than the local
            intrinsic dimension of the manifold.

        repulsion_strength: float (optional, default 1.0)
            Weighting applied to negative samples in low dimensional embedding
            optimization. Values higher than one will result in greater weight
            being given to negative samples.

        negative_sample_rate: int (optional, default 5)
            The number of negative samples to select per positive sample in the
            optimization process. Increasing this value will result in greater
            repulsive force being applied, greater optimization cost, but
            slightly more accuracy.

        a: float (optional, default None)
            More specific parameters controlling the embedding. If None these
            values are set automatically as determined by ``min_dist`` and
            ``spread``.

        b: float (optional, default None)
            More specific parameters controlling the embedding. If None these
            values are set automatically as determined by ``min_dist`` and
            ``spread``.

        random_state: int, RandomState instance or None, optional (default:
        None)
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.

        target_n_neighbors: int (optional, default -1)
            The number of nearest neighbors to use to construct the target
            simplicial set. If set to -1 use the ``n_neighbors`` value.

        target_metric: string or callable (optional, default 'categorical')
            The metric used to measure distance for a target array is using
            supervised dimension reduction. By default this is 'categorical'
            which will measure distance in terms of whether categories match or
            are different. Furthermore, if semi-supervised is required target
            values of -1 will be treated as unlabelled under the 'categorical'
            metric. If the target array takes continuous values (e.g. for a
            regression problem) then metric of 'l1' or 'l2' is probably more
            appropriate.

        target_metric_kwds: dict (optional, default None)
            Keyword argument to pass to the target metric when performing
            supervised dimension reduction. If None then no arguments are passed
            on.

        target_weight: float (optional, default 0.5)
            weighting factor between data topology and target topology. A value
            of 0.0 weights predominantly on data, a value of 1.0 places a strong
            emphasis on target. The default of 0.5 balances the weighting
            equally between data and target.

        transform_seed: int (optional, default 42)
            Random seed used for the stochastic aspects of the transform
            operation. This ensures consistency in transform operations.

        verbose: bool (optional, default False)
            Controls verbosity of logging.

        tqdm_kwds: dict (optional, default None)
            Key word arguments to be used by the tqdm progress bar.

        densmap: bool (optional, default False)
            Specifies whether the density-augmented objective of densMAP should
            be used for optimization. Turning on this option generates an
            embedding where the local densities are encouraged to be correlated
            with those in the original space. Parameters below with the prefix
            'dens' further control the behavior of this extension.

        dens_lambda: float (optional, default 2.0)
            Controls the regularization weight of the density correlation term
            in densMAP. Higher values prioritize density preservation over the
            UMAP objective, and vice versa for values closer to zero. Setting
            this parameter to zero is equivalent to running the original UMAP
            algorithm.

        dens_frac: float (optional, default 0.3)
            Controls the fraction of epochs (between 0 and 1) where the
            density-augmented objective is used in densMAP. The first (1 -
            dens_frac) fraction of epochs optimize the original UMAP objective
            before introducing the density correlation term.

        dens_var_shift: float (optional, default 0.1)
            A small constant added to the variance of local radii in the
            embedding when calculating the density correlation objective to
            prevent numerical instability from dividing by a small number

        output_dens: float (optional, default False)
            Determines whether the local radii of the final embedding (an
            inverse measure of local density) are computed and returned in
            addition to the embedding. If set to True, local radii of the
            original data are also included in the output for comparison; the
            output is a tuple (embedding, original local radii, embedding local
            radii). This option can also be used when densmap=False to calculate
            the densities for UMAP embeddings.

        disconnection_distance: float (optional, default np.inf or maximal value
        for bounded distances)
            Disconnect any vertices of distance greater than or equal to
            disconnection_distance when approximating the manifold via our k-nn
            graph. This is particularly useful in the case that you have a
            bounded metric.  The UMAP assumption that we have a connected
            manifold can be problematic when you have points that are maximally
            different from all the rest of your data.  The connected manifold
            assumption will make such points have perfect similarity to a random
            set of other points.  Too many such points will artificially connect
            your space.

        Returns
        -------
        umap : UMAP
            The fitted UMAP model.

        References
        ----------
        .. [1] McInnes, L., Healy, J. and Melville, J., 2018. Umap: Uniform
            manifold approximation and projection for dimension reduction. arXiv
            preprint arXiv:1802.03426.
        """
        check_is_fitted(
            self,
            ["_graph", "_raw_data"],
            msg="You first need to fit the estimator before accessing member functions.",
        )

        # --- Operate on clean_data below!

        data = self._raw_data
        if not self._all_finite:
            data = self._raw_data[self.finite_index]

        with warn.catch_warnings():
            warn.filterwarnings(
                "ignore",
                category=UserWarning,
                module="umap.umap_",
                message=".*is not an NNDescent object.*",
            )
            umap = UMAP(
                n_components=n_components,
                n_neighbors=self.graph_neighbors_.shape[1],
                metric=self.metric,
                metric_kwds=self.metric_kwds,
                precomputed_knn=csr_to_neighbor_list(
                    self._graph.data, self._graph.indices, self._graph.indptr
                ),
                output_metric=output_metric,
                output_metric_kwds=output_metric_kwds,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                init=init,
                min_dist=min_dist,
                spread=spread,
                set_op_mix_ratio=set_op_mix_ratio,
                local_connectivity=local_connectivity,
                repulsion_strength=repulsion_strength,
                negative_sample_rate=negative_sample_rate,
                a=a,
                b=b,
                random_state=random_state,
                target_n_neighbors=target_n_neighbors,
                target_metric=target_metric,
                target_metric_kwds=target_metric_kwds,
                target_weight=target_weight,
                transform_seed=transform_seed,
                transform_mode=transform_mode,
                verbose=verbose,
                tqdm_kwds=tqdm_kwds,
                densmap=densmap,
                dens_lambda=dens_lambda,
                dens_frac=dens_frac,
                dens_var_shift=dens_var_shift,
                output_dens=output_dens,
                disconnection_distance=disconnection_distance,
            ).fit(data)

        return umap

    def tsne(
        self,
        *,
        n_components: int = 2,
        init: np.ndarray | spmatrix | Literal["random", "pca"] = "pca",
        learning_rate: float | Literal["auto"] = "auto",
        early_exaggeration: float = 12.0,
        min_grad_norm: float = 1e-7,
        max_iter: int = 1000,
        n_iter_without_progress: int = 300,
        method: Literal["barnes_hut", "exact"] = "barnes_hut",
        angle: float = 0.5,
        random_state: RandomState | int | None = None,
        verbose: int = 0,
    ):
        """Constructs and fits a TSNE model [1]_ to the kMST graph.

         Unlike HDBSCAN and HBCC, TSNE does not support infinite data. To ensure
         all TSNE's member functions work as expected, the TSNE model is NOT
         remapped to the infinite data after fitting. As a result, combining TSNE
         and HDBSCAN results need to consider the finite index: ```
             plt.scatter(*tsne.embedding_.T,
             c=hdbscan.labels_[multi_mst.finite_index])
         ```

         Parameters
         ----------
         n_components : int, default=2
             Dimension of the embedded space.

         init : {"random", "pca"} or array of shape (n_samples, n_components), default="pca"
             Initialization of embedding.

         learning_rate : float or "auto", default="auto"
             The learning rate for t-SNE is usually in the range [10.0, 1000.0].
             If the learning rate is too high, the data may look like a 'ball'
             with any point approximately equidistant from its nearest neighbors.
             If the learning rate is too low, most points may look compressed in
             a dense cloud with few outliers. If the cost function gets stuck in
             a bad local minimum increasing the learning rate may help.

         early_exaggeration : float, default=12.0
             Controls how tight natural clusters in the original space are in the
             embedded space and how much space will be between them. For larger
             values, the space between natural clusters will be larger in the
             embedded space. The choice of this parameter is not very critical.
             If the cost function increases during initial optimization, the
             early exaggeration factor or the learning rate might be too high.

         min_grad_norm : float, default=1e-7
             If the gradient norm is below this threshold, the optimization will
             be stopped.

         max_iter : int, default=1000
             Maximum number of iterations for the optimization. Should be at
             least 250.

        n_iter_without_progress : int, default=300
             Maximum number of iterations without progress before we abort the
             optimization, used after 250 initial iterations with early
             exaggeration. Note that progress is only checked every 50 iterations
             so this value is rounded to the next multiple of 50.

        method : {'barnes_hut', 'exact'}, default='barnes_hut'
            By default the gradient calculation algorithm uses Barnes-Hut
            approximation running in O(NlogN) time. method='exact'
            will run on the slower, but exact, algorithm in O(N^2) time. The
            exact algorithm should be used when nearest-neighbor errors need
            to be better than 3%. However, the exact method cannot scale to
            millions of examples.

         angle : float, default=0.5
             This is the trade-off between speed and accuracy for Barnes-Hut
             T-SNE. 'angle' is the angular size of a distant node as measured
             from a point. If this size is below 'angle' then it is used as a
             summary node of all points contained within it. This method is not
             very sensitive to changes in this parameter in the range of 0.2 -
             0.8. Angle less than 0.2 has quickly increasing computation time and
             angle greater 0.8 has quickly increasing error.

         random_state : int, RandomState instance or None, default=None
             Determines the random number generator. Pass an int for reproducible
             results across multiple function calls. Note that different
             initializations might result in different local minima of the cost
             function.

         verbose : int, default=0
             Verbosity level.

         Returns
         -------
         tsne : TSNE
             The fitted TSNE model.

         References
         ----------
         .. [1] van der Maaten, L., & Hinton, G. (2008). Visualizing data using
         t-SNE. Journal of Machine Learning Research.
        """
        check_is_fitted(
            self,
            ["_graph", "_raw_data"],
            msg="You first need to fit the estimator before accessing member functions.",
        )
        if method == "barnes_hut" and n_components > 3:
            raise ValueError(
                "'n_components' should be inferior to 4 for the "
                "barnes_hut algorithm as it relies on "
                "quad-tree or oct-tree."
            )

        # Extract raw data
        X = self._raw_data
        if not self._all_finite:
            X = self._raw_data[self.finite_index]
        X = X.astype(np.float32, copy=False)

        # Build the t-SNE model
        n_samples = self._graph.shape[0]
        random_state = check_random_state(random_state)
        tsne = TSNE(
            method=method,
            metric=self.metric,
            metric_params=self.metric_kwds,
            n_components=n_components,
            init=init,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            n_iter_without_progress=n_iter_without_progress,
            angle=angle,
            random_state=random_state,
            verbose=verbose,
        )

        # Configure parameters set in fit
        if learning_rate == "auto":
            tsne.learning_rate_ = X.shape[0] / tsne.early_exaggeration / 4
            tsne.learning_rate_ = np.maximum(tsne.learning_rate_, 50)
        else:
            tsne.learning_rate_ = tsne.learning_rate

        # Do the initialization
        if isinstance(init, np.ndarray):
            X_embedded = init
        elif init == "pca":
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            pca.set_output(transform="default")
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
        elif init == "random":
            X_embedded = 1e-4 * random_state.standard_normal(
                size=(n_samples, n_components)
            ).astype(np.float32)

        # Fit tSNE optimizer (run on csr matrix)
        tsne.graph_, tsne.perplexity = _joint_probabilities_csr(self._graph, verbose)
        tsne.embedding_ = tsne._tsne(
            tsne.graph_, max(n_components - 1, 1), n_samples, X_embedded=X_embedded
        )
        return tsne

    def hdbscan(
        self,
        data_labels=None,
        sample_weights=None,
        *,
        min_cluster_size: int = 25,
        max_cluster_size: float = np.inf,
        allow_single_cluster: bool = False,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        ss_algorithm: Literal["bc", "bc_simple"] = "bc",
    ):
        """Constructs and fits an HDBSCAN [1]_ model to the kMST graph.

        Parameters
        ----------
        data_labels : array-like, shape (n_samples,), optional (default=None)
            Labels for semi-supervised clustering. If provided, the model will
            be semi-supervised and will use the provided labels to guide the
            clustering process.

        sample_weights : array-like, shape (n_samples,), optional (default=None)
            Data point weights used to adapt cluster size.

        min_cluster_size : int, optional (default=5)
            The minimum size of clusters; single linkage splits that contain
            fewer points than this will be considered points "falling out" of a
            cluster rather than a cluster splitting into two new clusters.

        cluster_selection_method : string, optional (default='eom')
            The method used to select clusters from the condensed tree. The
            standard approach for HDBSCAN* is to use an Excess of Mass algorithm
            to find the most persistent clusters. Alternatively you can instead
            select the clusters at the leaves of the tree -- this provides the
            most fine grained and homogeneous clusters. Options are:
                * ``eom``
                * ``leaf``

        allow_single_cluster : bool, optional (default=False)
            By default HDBSCAN* will not produce a single cluster, setting this
            to True will override this and allow single cluster results in the
            case that you feel this is a valid result for your dataset.

        cluster_selection_epsilon: float, optional (default=0.0)
            A distance threshold. Clusters below this value will be merged. This
            is the minimum epsilon allowed.

        cluster_selection_persistence: float, optional (default=0.0)
            A persistence threshold. Clusters with a persistence lower than this
            value will be merged.

        ss_algorithm: string, optional (default='bc')
            The semi-supervised clustering algorithm to use. Valid options are:
                * ``bc``
                * ``bc_simple``

        Returns
        -------
        clusterer : HDBSCAN
            The fitted HDBSCAN model.

        References
        ----------
        .. [1] McInnes L., Healy J. 2017. Accelerated Hierarchical Density Based
        Clustering. IEEE International Conference on Data Mining Workshops
        (ICDMW), pp 33-42.
        http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8215642.
        """
        check_is_fitted(
            self,
            ["_graph", "_raw_data", "minimum_spanning_tree_"],
            msg="You first need to fit the estimator before accessing member functions.",
        )
        validate_hdbscan_params(
            self._raw_data.shape[0],
            data_labels=data_labels,
            sample_weights=sample_weights,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            ss_algorithm=ss_algorithm,
        )
        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights, self._raw_data, dtype=np.float32
            )
        clusterer = HDBSCAN(
            min_samples=self.num_neighbors,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            semi_supervised=data_labels is not None,
            ss_algorithm=ss_algorithm,
        )

        # --- Operate on clean_data below!

        if not self._all_finite:
            if data_labels is not None:
                data_labels = data_labels[self.finite_index]
            if sample_weights is not None:
                sample_weights = sample_weights[self.finite_index]
        set_internals(
            clusterer,
            self._raw_data,
            self._knn_neighbors,
            self._knn_distances.T[-1],
            *clusters_from_spanning_tree(
                self.mutual_reachability_tree_,
                data_labels=data_labels,
                sample_weights=sample_weights,
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
                allow_single_cluster=allow_single_cluster,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_persistence=cluster_selection_persistence,
                semi_supervised=data_labels is not None,
                ss_algorithm=ss_algorithm,
            ),
        )

        # --- Map back to raw_data!

        if not self._all_finite:
            remap_hdbscan(
                clusterer,
                self.finite_index,
                self.internal_to_raw,
                self._raw_data.shape[0],
            )

        return clusterer

    def hbcc(
        self,
        data_labels=None,
        sample_weights=None,
        *,
        num_hops: int = 2,
        min_cluster_size: int = 25,
        max_cluster_size: float = np.inf,
        hop_type: Literal["manifold", "metric"] = "manifold",
        boundary_connectivity: Literal["knn", "core"] = "knn",
        boundary_use_reachability: bool = True,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        allow_single_cluster: bool = False,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        ss_algorithm: Literal["bc", "bc_simple"] = "bc",
    ):
        """Constructs and fits an HBCC model to the kMST graph.

        Parameters
        ----------
        data_labels : array-like, shape (n_samples,), optional (default=None)
            Labels for semi-supervised clustering. If provided, the model will
            be semi-supervised and will use the provided labels to guide the
            clustering process.

        sample_weights : array-like, shape (n_samples,), optional (default=None)
            Data point weights used to adapt cluster size.

        num_hops: int, default=2
            The number of hops used to expand the boundary coefficient
            connectivity graph.

        hop_type: 'manifold' or 'metric', default='manifold'
            The type of hop expansion to use. Manifold adds edge distances on
            traversal, metric computes distance between visited points.

        boundary_connectivity: 'knn' or 'core', default='knn'
            Which graph to compute the boundary coefficient on. 'knn' uses the
            k-nearest neighbors graph, 'core' uses the knn--mst union graph.

        boundary_use_reachability: boolean, default=False
            Whether to use mutual reachability or raw distances for the boundary
            coefficient computation.

        min_cluster_size : int, optional (default=5)
            The minimum size of clusters; single linkage splits that contain
            fewer points than this will be considered points "falling out" of a
            cluster rather than a cluster splitting into two new clusters.

        cluster_selection_method : string, optional (default='eom')
            The method used to select clusters from the condensed tree. The
            standard approach for HDBSCAN* is to use an Excess of Mass algorithm
            to find the most persistent clusters. Alternatively you can instead
            select the clusters at the leaves of the tree -- this provides the
            most fine grained and homogeneous clusters. Options are:
                * ``eom``
                * ``leaf``

        allow_single_cluster : bool, optional (default=False)
            By default HDBSCAN* will not produce a single cluster, setting this
            to True will override this and allow single cluster results in the
            case that you feel this is a valid result for your dataset.

        cluster_selection_epsilon: float, optional (default=0.0)
            A distance threshold. Clusters below this value will be merged. This
            is the minimum epsilon allowed.

        cluster_selection_persistence: float, optional (default=0.0)
            A persistence threshold. Clusters with a persistence lower than this
            value will be merged.

        ss_algorithm: string, optional (default='bc')
            The semi-supervised clustering algorithm to use. Valid options are:
                * ``bc``
                * ``bc_simple``

        Returns
        -------
        clusterer : HDBSCAN
            The fitted HDBSCAN model.
        """
        check_is_fitted(
            self,
            ["_graph", "_raw_data"],
            msg="You first need to fit the estimator before accessing member functions.",
        )
        validate_hdbscan_params(
            self._raw_data.shape[0],
            data_labels=data_labels,
            sample_weights=sample_weights,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            ss_algorithm=ss_algorithm,
        )
        validate_hbcc_params(
            num_hops=num_hops,
            hop_type=hop_type,
            boundary_connectivity=boundary_connectivity,
        )
        if self.metric != "euclidean":
            if hop_type == "metric":
                raise ValueError(
                    'BoundaryClusterDetector requires "euclidean" '
                    'metric with `hop_type="manifold".'
                )
            if not boundary_use_reachability:
                raise ValueError(
                    'BoundaryClusterDetector requires "euclidean" '
                    "metric with `boundary_use_reachability=False`."
                )

        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights, self._raw_data, dtype=np.float32
            )
        clusterer = HBCC(
            num_hops=num_hops,
            hop_type=hop_type,
            boundary_connectivity=boundary_connectivity,
            boundary_use_reachability=boundary_use_reachability,
            min_samples=self.num_neighbors,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            ss_algorithm=ss_algorithm,
        )

        # --- Operate on clean_data below!

        data = self._raw_data
        if not self._all_finite:
            data = data[self.finite_index]
            if data_labels is not None:
                data_labels = data_labels[self.finite_index]
            if sample_weights is not None:
                sample_weights = sample_weights[self.finite_index]

        if boundary_connectivity == "core":
            minimum_spanning_tree = self.mutual_reachability_tree_
        else:
            minimum_spanning_tree = np.empty((0, 3), dtype=np.float64)
        boundary_coefficient = compute_boundary_coefficient(
            data,
            self._knn_neighbors,
            self._knn_distances[:, -1],
            minimum_spanning_tree,
            num_hops=num_hops,
            hop_type=hop_type,
            boundary_connectivity=boundary_connectivity,
            boundary_use_reachability=boundary_use_reachability,
        )

        edges, graph = lensed_spanning_tree_from_graph(
            # cannot use mutual_reachability_graph_ here, as it includes infinite points
            csr_mutual_reachability(self._graph.copy(), self._knn_distances.T[-1]),
            boundary_coefficient,
        )
        set_internals(
            clusterer,
            self._raw_data,
            self._knn_neighbors,
            self._knn_distances.T[-1],
            *clusters_from_spanning_tree(
                edges,
                data_labels=data_labels,
                sample_weights=sample_weights,
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
                allow_single_cluster=allow_single_cluster,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_persistence=cluster_selection_persistence,
                semi_supervised=data_labels is not None,
                ss_algorithm=ss_algorithm,
            ),
        )
        clusterer.boundary_coefficient_ = boundary_coefficient
        clusterer._core_graph = graph

        # --- Map back to raw_data!

        if not self._all_finite:
            remap_hbcc(
                clusterer,
                self.finite_index,
                self.internal_to_raw,
                self._raw_data.shape[0],
            )

        return clusterer

    def branch_detector(
        self,
        clusterer,
        cluster_labels=None,
        cluster_probabilities=None,
        sample_weights=None,
        *,
        label_sides_as_branches: bool = False,
        min_cluster_size: int | None = None,
        max_cluster_size: int | None = None,
        allow_single_cluster: bool | None = None,
        cluster_selection_method: Literal["eom", "leaf"] | None = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        propagate_labels: bool = False,
    ):
        """Constructs and fits a metric-aware BranchDetector [1]_, ensuring
        valid parameter--metric combinations.

        Parameters
        ----------
        clusterer : HDBSCAN | HBCC
            The fitted HDBSCAN or HBCC model to use for branch detection.

        cluster_labels : np.ndarray, shape (n_samples, ), optional (default=None)
            Override cluster labels for each point in the data set. If not
            provided, the clusterer's labels will be used. Clusters must be
            connected in the minimum spanning tree. Otherwise, the branch
            detector will return connected component labels for that cluster.

        cluster_probabilities : np.ndarray, shape (n_samples, ), optional (default=None)
            Override cluster probabilities for each point in the data set. If
            not provided, the clusterer's probabilities will be used, or all
            points will be given 1.0 probability if cluster_labels are overridden.

        sample_weights : np.ndarray, shape (n_samples, ), optional (default=None)
            Data point weights used to adapt cluster size.

        label_sides_as_branches: bool, default=False
            Controls the minimum number of branches in a cluster for the
            branches to be labelled. When True, the branches are labelled if
            there are more than one branch in a cluster. When False, the
            branches are labelled if there are more than two branches in a
            cluster.

        min_cluster_size : int, optional (default=5)
            The minimum size of clusters; single linkage splits that contain
            fewer points than this will be considered points "falling out" of a
            cluster rather than a cluster splitting into two new clusters.

        allow_single_cluster : bool, optional (default=False)
            By default HDBSCAN* will not produce a single cluster, setting this
            to True will override this and allow single cluster results in the
            case that you feel this is a valid result for your dataset.

        cluster_selection_method : string, optional (default='eom')
            The method used to select clusters from the condensed tree. The
            standard approach for HDBSCAN* is to use an Excess of Mass algorithm
            to find the most persistent clusters. Alternatively you can instead
            select the clusters at the leaves of the tree -- this provides the
            most fine grained and homogeneous clusters. Options are:
                * ``eom``
                * ``leaf``

        cluster_selection_epsilon: float, optional (default=0.0)
            A distance threshold. Clusters below this value will be merged. This
            is the minimum epsilon allowed.

        cluster_selection_persistence: float, optional (default=0.0)
            A persistence threshold. Clusters with a persistence lower than this
            value will be merged.

        propagate_labels: bool, optional (default=False)
            Whether to fill in noise labels with (repeated) majority vote branch
            labels.

        Returns
        -------
        clusterer : BranchDetector
            A fitted BranchDetector.

        References
        ----------
        .. [1] Bot D.M., Peeters J., Liesenborgs J., Aerts J. 2025. FLASC: a
        flare-sensitive clustering algorithm. PeerJ Computer Science 11:e2792
        https://doi.org/10.7717/peerj-cs.2792.
        """

        return BranchDetector(
            metric=self.metric,
            metric_kwds=self.metric_kwds,
            label_sides_as_branches=label_sides_as_branches,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            propagate_labels=propagate_labels,
        ).fit(clusterer, cluster_labels, cluster_probabilities, sample_weights)

    def boundary_cluster_detector(
        self,
        clusterer,
        cluster_labels=None,
        cluster_probabilities=None,
        sample_weights=None,
        *,
        num_hops: int = 2,
        hop_type: Literal["manifold", "metric"] = "manifold",
        boundary_connectivity: Literal["knn", "core"] = "knn",
        boundary_use_reachability: bool = True,
        min_cluster_size: int | None = None,
        max_cluster_size: int | None = None,
        allow_single_cluster: bool | None = None,
        cluster_selection_method: Literal["eom", "leaf"] | None = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
    ):
        """Constructs a BoundaryClusterDetector, ensuring valid
        parameter--metric combinations.

        Parameters
        ----------
        clusterer : HDBSCAN | HBCC
            The fitted HDBSCAN or HBCC model to use for branch detection.

        cluster_labels : np.ndarray, shape (n_samples, ), optional (default=None)
            Override cluster labels for each point in the data set. If not
            provided, the clusterer's labels will be used. Clusters must be
            connected in the minimum spanning tree. Otherwise, the branch
            detector will return connected component labels for that cluster.

        cluster_probabilities : np.ndarray, shape (n_samples, ), optional (default=None)
            Override cluster probabilities for each point in the data set. If
            not provided, the clusterer's probabilities will be used, or all
            points will be given 1.0 probability if cluster_labels are overridden.

        sample_weights : np.ndarray, shape (n_samples, ), optional (default=None)
            Data point weights used to adapt cluster size.

        num_hops: int, default=2
            The number of hops used to expand the boundary coefficient
            connectivity graph.

        hop_type: 'manifold' or 'metric', default='manifold'
            The type of hop expansion to use. Manifold adds edge distances on
            traversal, metric computes distance between visited points.

        boundary_connectivity: 'knn' or 'core', default='knn'
            Which graph to compute the boundary coefficient on. 'knn' uses the
            k-nearest neighbors graph, 'core' uses the knn--mst union graph.

        boundary_use_reachability: boolean, default=False
            Whether to use mutual reachability or raw distances for the boundary
            coefficient computation.

        min_cluster_size : int, optional (default=5)
            The minimum size of clusters; single linkage splits that contain
            fewer points than this will be considered points "falling out" of a
            cluster rather than a cluster splitting into two new clusters.

        allow_single_cluster : bool, optional (default=False)
            By default HDBSCAN* will not produce a single cluster, setting this
            to True will override this and allow single cluster results in the
            case that you feel this is a valid result for your dataset.

        cluster_selection_method : string, optional (default='eom')
            The method used to select clusters from the condensed tree. The
            standard approach for HDBSCAN* is to use an Excess of Mass algorithm
            to find the most persistent clusters. Alternatively you can instead
            select the clusters at the leaves of the tree -- this provides the
            most fine grained and homogeneous clusters. Options are:
                * ``eom``
                * ``leaf``

        cluster_selection_epsilon: float, optional (default=0.0)
            A distance threshold. Clusters below this value will be merged. This
            is the minimum epsilon allowed.

        cluster_selection_persistence: float, optional (default=0.0)
            A persistence threshold. Clusters with a persistence lower than this
            value will be merged.

        Returns
        -------
        clusterer : BoundaryClusterDetector
            A fitted BoundaryClusterDetector.
        """

        if self.metric != "euclidean":
            if hop_type == "metric":
                raise ValueError(
                    'BoundaryClusterDetector requires "euclidean" '
                    'metric with `hop_type="manifold".'
                )
            if not boundary_use_reachability:
                raise ValueError(
                    'BoundaryClusterDetector requires "euclidean" '
                    "metric with `boundary_use_reachability=False`."
                )

        return BoundaryClusterDetector(
            num_hops=num_hops,
            hop_type=hop_type,
            boundary_connectivity=boundary_connectivity,
            boundary_use_reachability=boundary_use_reachability,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
        ).fit(clusterer, cluster_labels, cluster_probabilities, sample_weights)

    def graphviz_layout(self, prog="sfdp", **kwargs):
        """
        Computes a layout for the graph using Graphviz.

        Requires networkx and (py)graphviz to be installed and accessible on the
        system path.

        Parameters
        ----------
        prog : str
            The graphviz program to run.
        **kwargs
            Additional arguments to `networkx.nx_agraph.graphviz_layout`.
        
        Returns
        -------
        coords : ndarray of shape (num_points, 2)
            The coordinates of the nodes in the graph.
        """
        check_is_fitted(
            self,
            ["_graph"],
            msg="You first need to fit the estimator before accessing member functions.",
        )
        import networkx as nx

        g = nx.Graph(self.graph_)
        pos = nx.nx_agraph.graphviz_layout(g, prog=prog, **kwargs)
        coords = np.nan * np.ones((self.graph_.shape[0], 2), dtype=np.float64)
        for k, v in pos.items():
            coords[k, :] = v
        return coords


def set_internals(
    clusterer,
    raw_data,
    neighbors,
    core_distances,
    labels,
    probability,
    linkage_tree,
    condensed_tree,
    spanning_tree,
):
    clusterer.labels_ = labels
    clusterer.probabilities_ = probability
    clusterer._neighbors = neighbors
    clusterer._core_distances = core_distances
    clusterer._min_spanning_tree = spanning_tree
    clusterer._single_linkage_tree = linkage_tree
    clusterer._condensed_tree = to_numpy_rec_array(condensed_tree)
    clusterer._raw_data = raw_data


def validate_hdbscan_params(
    num_points,
    data_labels=None,
    sample_weights=None,
    min_cluster_size=10,
    max_cluster_size=np.inf,
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    ss_algorithm="bc",
):
    check_greater_equal(
        min_value=2,
        datatype=np.integer,
        min_cluster_size=min_cluster_size,
    )
    max_cluster_size = float(max_cluster_size)
    check_greater_equal(
        min_value=0.0,
        datatype=np.floating,
        max_cluster_size=max_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_persistence=cluster_selection_persistence,
    )
    check_optional_size(
        num_points, data_labels=data_labels, sample_weights=sample_weights
    )
    check_literals(
        cluster_selection_method=(cluster_selection_method, ["eom", "leaf"]),
        ss_algorithm=(ss_algorithm, ["bc", "bc_simple"]),
    )


def validate_hbcc_params(num_hops=2, hop_type="manifold", boundary_connectivity="knn"):
    check_greater_equal(
        min_value=0,
        datatype=np.integer,
        num_hops=num_hops,
    )
    check_literals(
        hop_type=(hop_type, ["manifold", "metric"]),
        boundary_connectivity=(boundary_connectivity, ["knn", "core"]),
    )


def remap_hdbscan(clusterer, finite_index, internal_to_raw, num_points):
    outliers = list(set(range(num_points)) - set(finite_index))
    clusterer._condensed_tree = remap_condensed_tree(
        clusterer._condensed_tree, internal_to_raw, outliers
    )
    clusterer._single_linkage_tree = remap_single_linkage_tree(
        clusterer._single_linkage_tree, internal_to_raw, outliers
    )

    new_labels = np.full(num_points, -1)
    new_labels[finite_index] = clusterer.labels_
    clusterer.labels_ = new_labels

    new_probabilities = np.zeros(num_points)
    new_probabilities[finite_index] = clusterer.probabilities_
    clusterer.probabilities_ = new_probabilities


def remap_hbcc(clusterer, finite_index, internal_to_raw, num_points):
    remap_hdbscan(clusterer, finite_index, internal_to_raw, num_points)
    clusterer._core_graph = remap_core_graph(
        clusterer._core_graph,
        finite_index,
        internal_to_raw,
        num_points,
    )
    new_bc = np.zeros(num_points)
    new_bc[finite_index] = clusterer.boundary_coefficient_
    clusterer.boundary_coefficient_ = new_bc


@numba.njit()
def _binary_search_perplexity_csr(dists, indptr):
    INFINITY = np.inf
    EPSILON_DBL = 1e-8
    PERPLEXITY_TOLERANCE = 1e-5
    n_steps = 100
    desired_perplexity = (np.diff(indptr).max() - 1) / 3
    desired_entropy = math.log(desired_perplexity)

    probs = np.empty_like(dists)
    for start, end in zip(indptr[:-1], indptr[1:]):
        beta_min = -INFINITY
        beta_max = INFINITY
        beta = 1.0

        # Binary search for beta to achieve desired perplexity
        for _ in range(n_steps):
            # Convert distances to similarities
            sum_Pi = 0.0
            for idx in range(start, end):
                probs[idx] = math.exp(-dists[idx] * beta)
                sum_Pi += probs[idx]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL

            # Normalize the probabilities
            sum_disti_Pi = 0.0
            for idx in range(start, end):
                probs[idx] /= sum_Pi
                sum_disti_Pi += dists[idx] * probs[idx]

            # Compute the resulting entropy
            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy
            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            # Update beta values
            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

    return probs, desired_perplexity


def _joint_probabilities_csr(graph, verbose):
    """Compute joint probabilities p_ij from sparse distances."""
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    graph.sort_indices()
    n_samples = graph.shape[0]
    conditional_P, perplexity = _binary_search_perplexity_csr(graph.data, graph.indptr)
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_array(
        (conditional_P, graph.indices, graph.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), np.finfo(np.double).eps)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return P, perplexity
