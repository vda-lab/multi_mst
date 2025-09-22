"""Tests for k-MST"""

import pytest
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_array

from umap import UMAP
from sklearn.manifold import TSNE
from fast_hbcc import HBCC, BoundaryClusterDetector
from fast_hdbscan import HDBSCAN
from multi_mst import kMST, KMST
from multi_mst.lib import BranchDetector


def invariants(p, X):
    g = p.graph_
    assert g.shape == (X.shape[0], X.shape[0])
    assert connected_components(g, directed=False, return_labels=False) == 1
    for point, (start, end) in enumerate(zip(g.indptr[:-1], g.indptr[1:])):
        assert np.all(g.indices[start:end] != point)
        assert np.all(g.data[start:end] > 0.0)

    mrg = p.mutual_reachability_graph_
    assert mrg.shape == g.shape
    assert connected_components(mrg, directed=False, return_labels=False) == 1
    for point, (start, end) in enumerate(zip(mrg.indptr[:-1], mrg.indptr[1:])):
        assert np.all(mrg.indices[start:end] != point)
        assert np.all(
            np.isclose(mrg.data[start:end], p.knn_distances_[point, -1])
            | (mrg.data[start:end] > p.knn_distances_[point, -1])
        )

    mst = p.minimum_spanning_tree_
    mst_g = coo_array(
        (mst[:, 2], (mst[:, 0].astype(np.int32), mst[:, 1].astype(np.int32))),
        g.shape,
    )
    assert mst.shape[0] == (X.shape[0] - 1)
    assert np.all(mst[:, 2] > 0.0)
    assert connected_components(mst_g, directed=False, return_labels=False) == 1

    mrmst = p.mutual_reachability_tree_
    mrmst_g = coo_array(
        (mrmst[:, 2], (mrmst[:, 0].astype(np.int32), mrmst[:, 1].astype(np.int32))),
        g.shape,
    )
    assert mrmst.shape[0] == (X.shape[0] - 1)
    assert connected_components(mrmst_g, directed=False, return_labels=False) == 1
    assert not np.any(mrmst[:, 2] < mst[:, 2])

    assert p.knn_neighbors_.shape == (X.shape[0], p.num_neighbors)
    assert p.knn_distances_.shape == (X.shape[0], p.num_neighbors)
    assert np.all(p.knn_neighbors_[:, 0] != np.arange(X.shape[0]))
    assert np.all(p.knn_distances_ > 0.0)
    assert np.all(np.diff(p.knn_distances_, axis=-1) >= 0.0)

    assert p.graph_neighbors_.shape[0] == X.shape[0]
    assert p.graph_neighbors_.shape[1] >= p.num_neighbors
    assert p.graph_distances_.shape[0] == X.shape[0]
    assert p.graph_distances_.shape[1] >= p.num_neighbors
    assert np.all(p.graph_neighbors_[:, 0] != np.arange(X.shape[0]))
    assert np.all(p.graph_distances_[p.graph_neighbors_ != -1] > 0.0)
    for row in range(X.shape[0]):
        assert np.all(
            np.diff(p.graph_distances_[row, p.graph_neighbors_[row, :] != -1]) >= 0.0
        )


def test_badargs(X):
    """Tests parameter validation."""
    with pytest.raises(ValueError):
        kMST(X, num_neighbors=1.0)
    with pytest.raises(ValueError):
        kMST(X, num_neighbors=-1)
    with pytest.raises(ValueError):
        kMST(X, epsilon=1)
    with pytest.raises(ValueError):
        kMST(X, epsilon=0.6)
    with pytest.raises(ValueError):
        kMST(X, epsilon=-0.4)
    with pytest.raises(ValueError):
        kMST(X, min_samples=1.0)
    with pytest.raises(ValueError):
        kMST(X, num_neighbors=3, min_samples=4)
    with pytest.raises(ValueError):
        kMST(X, min_samples=0)
    with pytest.raises(ValueError):
        kMST(X, min_samples=-1)


def test_defaults(X):
    """Tests with default parameters."""
    p = KMST().fit(X)
    invariants(p, X)


def test_min_samples(X):
    """Tests with higher min_samples."""
    p = KMST(min_samples=3).fit(X)
    invariants(p, X)


def test_num_neighbors(X):
    """Tests with lower num_neighbors."""
    p = KMST(num_neighbors=1).fit(X)
    invariants(p, X)


def test_epsilon(X):
    """Tests with epsilon."""
    base = KMST().fit(X)
    p = KMST(epsilon=1.1).fit(X)
    invariants(p, X)
    assert p.graph_.nnz < base.graph_.nnz


def test_with_missing_data(X_missing, clean_indices):
    """Tests with nan data."""
    model = KMST().fit(X_missing)
    clean_model = KMST().fit(X_missing[clean_indices])

    # No edges to the missing data
    assert np.all(
        clean_model.graph_.indptr[1:]
        == model.graph_.indptr[np.array(clean_indices) + 1]
    )
    assert np.all(model.graph_.indices != 0)
    assert model.graph_.indptr[1] == model.graph_.indptr[0]
    assert np.all(model.graph_neighbors_[0, :] == -1)
    assert np.isinf(model.graph_distances_[0, :]).all()

    assert np.all(model.graph_.indices != 6)
    assert model.graph_.indptr[7] == model.graph_.indptr[6]
    assert np.all(model.graph_neighbors_[6, :] == -1)
    assert np.isinf(model.graph_distances_[6, :]).all()

    # Other edges should be the same
    assert np.allclose(clean_model.graph_.data, model.graph_.data)
    assert np.allclose(
        clean_model.graph_distances_, model.graph_distances_[clean_indices]
    )


def test_umap(X_missing):
    model = KMST().fit(X_missing)
    umap = model.umap()

    assert isinstance(umap, UMAP)
    assert umap.embedding_.shape == (X_missing.shape[0] - 2, 2)


def test_tsne(X_missing):
    model = KMST().fit(X_missing)
    tsne = model.tsne()

    assert isinstance(tsne, TSNE)
    assert tsne.embedding_.shape == (X_missing.shape[0] - 2, 2)


def test_hdbscan(X_missing):
    model = KMST().fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == X_missing.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == X_missing.shape[0]
    assert hdbscan.labels_[0] == -1
    assert hdbscan.labels_[6] == -1
    assert len(set(hdbscan.labels_)) == 6


def test_hdbscan_min_samples(X_missing):
    model = KMST(min_samples=3).fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == X_missing.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == X_missing.shape[0]
    assert hdbscan.labels_[0] == -1
    assert hdbscan.labels_[6] == -1
    assert len(set(hdbscan.labels_)) == 6


def test_hdbscan_weighted(X_missing):
    model = KMST().fit(X_missing)
    hdbscan = model.hdbscan(
        sample_weights=np.ones(X_missing.shape[0]) / 2, min_cluster_size=5
    )

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == X_missing.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == X_missing.shape[0]
    assert hdbscan.labels_[0] == -1
    assert hdbscan.labels_[6] == -1
    assert len(set(hdbscan.labels_)) == 3


def test_hdbscan_branches(X_missing):
    model = KMST().fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)
    d = model.branch_detector(hdbscan)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) == 9


def test_hdbscan_boundary_clusters(X_missing):
    model = KMST().fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan)

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 6


def test_hbcc(X_missing):
    model = KMST().fit(X_missing)
    hbcc = model.hbcc(min_cluster_size=5)

    assert isinstance(hbcc, HBCC)
    assert hbcc.min_samples == model.num_neighbors
    assert hbcc._raw_data is model._raw_data
    assert hbcc._raw_data.shape[0] == X_missing.shape[0]
    assert hbcc._neighbors is model._knn_neighbors
    assert np.allclose(hbcc._core_distances, model._knn_distances[:, -1])
    assert hbcc.labels_.shape[0] == X_missing.shape[0]
    assert hbcc.labels_[0] == -1
    assert hbcc.labels_[6] == -1
    assert len(set(hbcc.labels_)) > 1


def test_hbcc_branches(X_missing):
    model = KMST().fit(X_missing)
    hbcc = model.hbcc(min_cluster_size=5)
    d = model.branch_detector(hbcc)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) > 1
