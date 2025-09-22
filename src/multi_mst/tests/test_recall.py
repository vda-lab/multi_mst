"""Tests for k-MST descent with recall"""

import pytest
import numpy as np

from umap import UMAP
from sklearn.manifold import TSNE
from fast_hbcc import HBCC, BoundaryClusterDetector
from fast_hdbscan import HDBSCAN
from multi_mst import KMSTDescentLogRecall, kMSTDescentLogRecall
from multi_mst.lib import BranchDetector

from .test_kmst import invariants


def test_badargs(X):
    """Tests parameter validation."""
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, num_neighbors=1.0)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, num_neighbors=-1)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, min_descent_neighbors=1.0)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, min_descent_neighbors=-1)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, epsilon=1)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, epsilon=0.6)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, epsilon=-0.4)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, min_samples=1.0)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, num_neighbors=3, min_samples=4)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, min_samples=0)
    with pytest.raises(ValueError):
        kMSTDescentLogRecall(X, min_samples=-1)


def test_defaults(X):
    """Tests with default parameters."""
    p = KMSTDescentLogRecall().fit(X)
    invariants(p, X)

    assert isinstance(p.trace_, list)
    assert len(p.trace_) > 0
    assert "boruvka_num_components" in p.trace_[0]
    assert "boruvka_recall" in p.trace_[0]
    assert "descent_recall" in p.trace_[0]
    assert "descent_distance_fraction" in p.trace_[0]
    assert "descent_num_changes" in p.trace_[0]
    assert p.trace_[0]["boruvka_num_components"] > 0
    assert p.trace_[0]["boruvka_recall"] > 0.0
    assert isinstance(p.trace_[0]["descent_recall"], np.ndarray)
    assert isinstance(p.trace_[0]["descent_distance_fraction"], np.ndarray)
    assert isinstance(p.trace_[0]["descent_num_changes"], np.ndarray)
    assert p.trace_[0]["descent_recall"][0] > 0.0
    assert p.trace_[0]["descent_distance_fraction"][0] > 0.0
    assert p.trace_[0]["descent_num_changes"][0] > 0


def test_min_samples(X):
    """Tests with higher min_samples."""
    p = KMSTDescentLogRecall(min_samples=3).fit(X)
    invariants(p, X)


def test_num_neighbors(X):
    """Tests with lower num_neighbors."""
    p = KMSTDescentLogRecall(num_neighbors=1).fit(X)
    invariants(p, X)


def test_descent_neighbors(X):
    """Tests with lower descent_neighbors."""
    p = KMSTDescentLogRecall(min_descent_neighbors=6).fit(X)
    invariants(p, X)


def test_epsilon(X):
    """Tests with epsilon."""
    base = KMSTDescentLogRecall().fit(X)
    p = KMSTDescentLogRecall(epsilon=1.1).fit(X)
    invariants(p, X)
    assert p.graph_.nnz < base.graph_.nnz


def test_with_missing_data(X_missing, clean_indices):
    """Tests with nan data."""
    model = KMSTDescentLogRecall().fit(X_missing)
    clean_model = KMSTDescentLogRecall().fit(X_missing[clean_indices])

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
    model = KMSTDescentLogRecall().fit(X_missing)
    umap = model.umap()

    assert isinstance(umap, UMAP)
    assert umap.embedding_.shape == (X_missing.shape[0] - 2, 2)


def test_tsne(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
    tsne = model.tsne()

    assert isinstance(tsne, TSNE)
    assert tsne.embedding_.shape == (X_missing.shape[0] - 2, 2)


def test_hdbscan(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
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
    model = KMSTDescentLogRecall(min_samples=3).fit(X_missing)
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
    model = KMSTDescentLogRecall().fit(X_missing)
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
    model = KMSTDescentLogRecall().fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)
    d = model.branch_detector(hdbscan)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) == 9


def test_hdbscan_boundary_clusters(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan)

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 6


def test_hdbscan_boundary_clusters_no_reachability(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan, boundary_use_reachability=False)

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 7


def test_hdbscan_boundary_clusters_metric(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan, hop_type="metric")

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 6


def test_hbcc(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
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


def test_hbcc_no_reachability(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
    hbcc = model.hbcc(min_cluster_size=5, boundary_use_reachability=False)

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


def test_hbcc_metric(X_missing):
    model = KMSTDescentLogRecall().fit(X_missing)
    hbcc = model.hbcc(min_cluster_size=5, hop_type="metric")

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
    model = KMSTDescentLogRecall().fit(X_missing)
    hbcc = model.hbcc(min_cluster_size=5)
    d = model.branch_detector(hbcc)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) > 1
