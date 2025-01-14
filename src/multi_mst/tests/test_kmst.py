"""Tests for k-MST"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_array

from umap import UMAP
from fast_hbcc import HBCC, BoundaryClusterDetector
from fast_hdbscan import HDBSCAN
from multi_mst import kMST, KMST
from multi_mst.lib import BranchDetector


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


def invariants(p):
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
        assert np.all(mrg.data[start:end] >= p.knn_distances_[point, -1])

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


def test_badargs():
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


def test_defaults():
    """Tests with default parameters."""
    p = KMST().fit(X)
    invariants(p)


def test_min_samples():
    """Tests with higher min_samples."""
    p = KMST(min_samples=3).fit(X)
    invariants(p)


def test_num_neighbors():
    """Tests with lower num_neighbors."""
    p = KMST(num_neighbors=1).fit(X)
    invariants(p)


def test_epsilon():
    """Tests with epsilon."""
    base = KMST().fit(X)
    p = KMST(epsilon=1.1).fit(X)
    invariants(p)
    assert p.graph_.nnz < base.graph_.nnz


def test_with_missing_data():
    """Tests with nan data."""
    model = KMST().fit(X_missing_data)
    clean_model = KMST().fit(X_missing_data[clean_indices])

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
    model = KMST().fit(X_missing_data)
    umap = model.umap()

    assert isinstance(umap, UMAP)
    assert umap.embedding_.shape == (X_missing_data.shape[0] - 2, 2)


def test_hdbscan():
    model = KMST().fit(X_missing_data)
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
    model = KMST(min_samples=3).fit(X_missing_data)
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
    model = KMST().fit(X_missing_data)
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
    model = KMST().fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)
    d = model.branch_detector(hdbscan)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) == 9


def test_hdbscan_boundary_clusters():
    model = KMST().fit(X_missing_data)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan)

    assert isinstance(bc, BoundaryClusterDetector)
    assert bc.labels_[0] == -1
    assert bc.labels_[6] == -1
    assert len(set(bc.labels_)) == 9


def test_hbcc():
    model = KMST().fit(X_missing_data)
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


def test_hbcc_branches():
    model = KMST().fit(X_missing_data)
    hbcc = model.hbcc(min_cluster_size=5)
    d = model.branch_detector(hbcc)

    assert isinstance(d, BranchDetector)
    assert d.labels_[0] == -1
    assert d.labels_[6] == -1
    assert len(set(d.labels_)) > 1
