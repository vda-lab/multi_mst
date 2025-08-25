"""Tests for k-MST"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from umap import UMAP
from fast_hbcc import HBCC, BoundaryClusterDetector
from fast_hdbscan import HDBSCAN
from multi_mst import kMSTDescent, KMSTDescent
from multi_mst.lib import BranchDetector

from .test_kmst import invariants


def generate_noisy_data():
    blobs, yBlobs = make_blobs(
        n_samples=50,
        centers=[(-0.75, 2.25), (2.0, -0.5)],
        cluster_std=0.2,
        random_state=3,
    )
    np.random.seed(5)
    noise = np.random.uniform(-1.0, 3.0, (50, 2))
    yNoise = np.full(50, -1)
    return (
        np.vstack((blobs, noise)),
        np.concatenate((yBlobs, yNoise)),
    )


X, y = generate_noisy_data()
X = StandardScaler().fit_transform(X)

clean_indices = list(range(1, 6)) + list(range(7, X.shape[0]))
X_missing_data = X.copy()
X_missing_data[0] = [np.nan, 1]
X_missing_data[6] = [np.nan, np.nan]


def test_badargs():
    """Tests parameter validation."""
    with pytest.raises(ValueError):
        kMSTDescent(X, num_neighbors=1.0)
    with pytest.raises(ValueError):
        kMSTDescent(X, num_neighbors=-1)
    with pytest.raises(ValueError):
        kMSTDescent(X, min_descent_neighbors=1.0)
    with pytest.raises(ValueError):
        kMSTDescent(X, min_descent_neighbors=-1)
    with pytest.raises(ValueError):
        kMSTDescent(X, epsilon=1)
    with pytest.raises(ValueError):
        kMSTDescent(X, epsilon=0.6)
    with pytest.raises(ValueError):
        kMSTDescent(X, epsilon=-0.4)
    with pytest.raises(ValueError):
        kMSTDescent(X, min_samples=1.0)
    with pytest.raises(ValueError):
        kMSTDescent(X, num_neighbors=3, min_samples=4)
    with pytest.raises(ValueError):
        kMSTDescent(X, min_samples=0)
    with pytest.raises(ValueError):
        kMSTDescent(X, min_samples=-1)


def test_defaults():
    """Tests with default parameters."""
    p = KMSTDescent().fit(X)
    invariants(p)


def test_min_samples():
    """Tests with higher min_samples."""
    p = KMSTDescent(min_samples=3).fit(X)
    invariants(p)


def test_num_neighbors():
    """Tests with lower num_neighbors."""
    p = KMSTDescent(num_neighbors=1).fit(X)
    invariants(p)


def test_descent_neighbors():
    """Tests with lower descent_neighbors."""
    p = KMSTDescent(min_descent_neighbors=6).fit(X)
    invariants(p)


def test_epsilon():
    """Tests with epsilon."""
    base = KMSTDescent().fit(X)
    p = KMSTDescent(epsilon=1.1).fit(X)
    invariants(p)
    assert p.graph_.nnz < base.graph_.nnz


def test_metric():
    """Tests with higher min_samples."""
    p = KMSTDescent(metric="sqeuclidean").fit(X)
    invariants(p)


def test_with_missing_data():
    """Tests with nan data."""
    model = KMSTDescent().fit(X_missing_data)
    clean_model = KMSTDescent().fit(X_missing_data[clean_indices])

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


def test_umap():
    model = KMSTDescent().fit(X_missing_data)
    umap = model.umap()

    assert isinstance(umap, UMAP)
    assert umap.embedding_.shape == (X_missing_data.shape[0] - 2, 2)


def test_hdbscan():
    model = KMSTDescent().fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == X_missing_data.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == X_missing_data.shape[0]
    assert hdbscan.labels_[0] == -1
    assert hdbscan.labels_[6] == -1
    assert len(set(hdbscan.labels_)) == 6


def test_hdbscan_min_samples():
    model = KMSTDescent(min_samples=3).fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == X_missing_data.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == X_missing_data.shape[0]
    assert hdbscan.labels_[0] == -1
    assert hdbscan.labels_[6] == -1
    assert len(set(hdbscan.labels_)) == 6


def test_hdbscan_weighted():
    model = KMSTDescent(metric="cosine").fit(X_missing_data)
    hdbscan = model.hdbscan(sample_weights=y, min_cluster_size=5)

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == X_missing_data.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == X_missing_data.shape[0]
    assert hdbscan.labels_[0] == -1
    assert hdbscan.labels_[6] == -1
    assert len(set(hdbscan.labels_)) == 1


def test_hdbscan_branches():
    model = KMSTDescent(metric="cosine").fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)
    d = model.branch_detector(hdbscan)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) == 8


def test_hdbscan_boundary_clusters():
    model = KMSTDescent().fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan)

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 6


def test_hdbscan_boundary_clusters_no_reachability():
    model = KMSTDescent().fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan, boundary_use_reachability=False)

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 7


def test_hdbscan_boundary_clusters_metric():
    model = KMSTDescent().fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan, hop_type="metric")

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 6


def test_hdbscan_boundary_clusters_not_euclidean():
    model = KMSTDescent(metric="cosine").fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan)

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 9

    with pytest.raises(ValueError):
        model.boundary_cluster_detector(hdbscan, boundary_use_reachability=False)

    with pytest.raises(ValueError):
        model.boundary_cluster_detector(hdbscan, hop_type="metric")


def test_hbcc():
    model = KMSTDescent().fit(X_missing_data)
    hbcc = model.hbcc(min_cluster_size=5)

    assert isinstance(hbcc, HBCC)
    assert hbcc.min_samples == model.num_neighbors
    assert hbcc._raw_data is model._raw_data
    assert hbcc._raw_data.shape[0] == X_missing_data.shape[0]
    assert hbcc._neighbors is model._knn_neighbors
    assert np.allclose(hbcc._core_distances, model._knn_distances[:, -1])
    assert hbcc.labels_.shape[0] == X_missing_data.shape[0]
    assert hbcc.labels_[0] == -1
    assert hbcc.labels_[6] == -1
    assert len(set(hbcc.labels_)) > 1


def test_hbcc_no_reachability():
    model = KMSTDescent().fit(X_missing_data)
    hbcc = model.hbcc(min_cluster_size=5, boundary_use_reachability=False)

    assert isinstance(hbcc, HBCC)
    assert hbcc.min_samples == model.num_neighbors
    assert hbcc._raw_data is model._raw_data
    assert hbcc._raw_data.shape[0] == X_missing_data.shape[0]
    assert hbcc._neighbors is model._knn_neighbors
    assert np.allclose(hbcc._core_distances, model._knn_distances[:, -1])
    assert hbcc.labels_.shape[0] == X_missing_data.shape[0]
    assert hbcc.labels_[0] == -1
    assert hbcc.labels_[6] == -1
    assert len(set(hbcc.labels_)) > 1


def test_hbcc_metric():
    model = KMSTDescent().fit(X_missing_data)
    hbcc = model.hbcc(min_cluster_size=5, hop_type="metric")

    assert isinstance(hbcc, HBCC)
    assert hbcc.min_samples == model.num_neighbors
    assert hbcc._raw_data is model._raw_data
    assert hbcc._raw_data.shape[0] == X_missing_data.shape[0]
    assert hbcc._neighbors is model._knn_neighbors
    assert np.allclose(hbcc._core_distances, model._knn_distances[:, -1])
    assert hbcc.labels_.shape[0] == X_missing_data.shape[0]
    assert hbcc.labels_[0] == -1
    assert hbcc.labels_[6] == -1
    assert len(set(hbcc.labels_)) > 1


def test_hbcc_not_euclidean():
    model = KMSTDescent(metric="cosine").fit(X_missing_data)
    hbcc = model.hbcc(min_cluster_size=5)

    assert isinstance(hbcc, HBCC)
    assert hbcc.min_samples == model.num_neighbors
    assert hbcc._raw_data is model._raw_data
    assert hbcc._raw_data.shape[0] == X_missing_data.shape[0]
    assert hbcc._neighbors is model._knn_neighbors
    assert np.allclose(hbcc._core_distances, model._knn_distances[:, -1])
    assert hbcc.labels_.shape[0] == X_missing_data.shape[0]
    assert hbcc.labels_[0] == -1
    assert hbcc.labels_[6] == -1
    assert len(set(hbcc.labels_)) > 1

    with pytest.raises(ValueError):
        model.hbcc(boundary_use_reachability=False)

    with pytest.raises(ValueError):
        model.hbcc(hop_type="metric")


def test_hbcc_branches():
    model = KMSTDescent(metric="cosine").fit(X_missing_data)
    hbcc = model.hbcc(min_cluster_size=5)
    d = model.branch_detector(hbcc)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) > 1
