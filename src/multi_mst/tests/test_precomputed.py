"""Tests for k-MST descent with precomputed distances"""

import pytest
import numpy as np

from umap import UMAP
from sklearn.manifold import TSNE
from fast_hbcc import HBCC, BoundaryClusterDetector
from fast_hdbscan import HDBSCAN
from multi_mst import KMSTDescent

from .test_kmst import invariants


def test_defaults(distance_matrix):
    """Tests with default parameters."""
    p = KMSTDescent(metric="precomputed").fit(distance_matrix)
    invariants(p, distance_matrix)


def test_umap(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    umap = model.umap()

    assert isinstance(umap, UMAP)
    assert umap.embedding_.shape == (distance_matrix.shape[0], 2)


def test_tsne(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    with pytest.raises(ValueError):
        model.tsne()

    tsne = model.tsne(init="random")
    assert isinstance(tsne, TSNE)
    assert tsne.embedding_.shape == (distance_matrix.shape[0], 2)


def test_hdbscan(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(min_cluster_size=5)

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == distance_matrix.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == distance_matrix.shape[0]
    assert len(set(hdbscan.labels_)) == 6


def test_hdbscan_min_samples(distance_matrix):
    model = KMSTDescent(min_samples=3, metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(min_cluster_size=5)

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == distance_matrix.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == distance_matrix.shape[0]
    assert len(set(hdbscan.labels_)) == 6


def test_hdbscan_weighted(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(
        sample_weights=np.ones(distance_matrix.shape[0]) / 2, min_cluster_size=5
    )

    assert isinstance(hdbscan, HDBSCAN)
    assert hdbscan.min_samples == model.num_neighbors
    assert hdbscan._raw_data is model._raw_data
    assert hdbscan._raw_data.shape[0] == distance_matrix.shape[0]
    assert hdbscan._neighbors is model._knn_neighbors
    assert np.allclose(hdbscan._core_distances, model._knn_distances[:, -1])
    assert hdbscan.labels_.shape[0] == distance_matrix.shape[0]
    assert len(set(hdbscan.labels_)) == 4


def test_hdbscan_branches(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(min_cluster_size=5)
    with pytest.raises(ValueError):
        model.branch_detector(hdbscan)


def test_hdbscan_boundary_clusters(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan)

    assert isinstance(bc, BoundaryClusterDetector)
    assert len(set(bc.labels_)) == 6


def test_hdbscan_boundary_clusters_no_reachability(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(min_cluster_size=5)
    with pytest.raises(ValueError):
        model.boundary_cluster_detector(hdbscan, boundary_use_reachability=False)


def test_hdbscan_boundary_clusters_metric(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(min_cluster_size=5)
    with pytest.raises(ValueError):
        model.boundary_cluster_detector(hdbscan, hop_type="metric")


def test_hdbscan_boundary_clusters_not_euclidean(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hdbscan = model.hdbscan(min_cluster_size=5)
    bc = model.boundary_cluster_detector(hdbscan)

    assert isinstance(bc, BoundaryClusterDetector)
    assert len(set(bc.labels_)) == 6

    with pytest.raises(ValueError):
        model.boundary_cluster_detector(hdbscan, boundary_use_reachability=False)

    with pytest.raises(ValueError):
        model.boundary_cluster_detector(hdbscan, hop_type="metric")


def test_hbcc(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hbcc = model.hbcc(min_cluster_size=5)

    assert isinstance(hbcc, HBCC)
    assert hbcc.min_samples == model.num_neighbors
    assert hbcc._raw_data is model._raw_data
    assert hbcc._raw_data.shape[0] == distance_matrix.shape[0]
    assert hbcc._neighbors is model._knn_neighbors
    assert np.allclose(hbcc._core_distances, model._knn_distances[:, -1])
    assert hbcc.labels_.shape[0] == distance_matrix.shape[0]
    assert len(set(hbcc.labels_)) > 1


def test_hbcc_no_reachability(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    with pytest.raises(ValueError):
        model.hbcc(min_cluster_size=5, boundary_use_reachability=False)


def test_hbcc_metric(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    with pytest.raises(ValueError):
        model.hbcc(min_cluster_size=5, hop_type="metric")


def test_hbcc_not_euclidean(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hbcc = model.hbcc(min_cluster_size=5)

    assert isinstance(hbcc, HBCC)
    assert hbcc.min_samples == model.num_neighbors
    assert hbcc._raw_data is model._raw_data
    assert hbcc._raw_data.shape[0] == distance_matrix.shape[0]
    assert hbcc._neighbors is model._knn_neighbors
    assert np.allclose(hbcc._core_distances, model._knn_distances[:, -1])
    assert hbcc.labels_.shape[0] == distance_matrix.shape[0]
    assert len(set(hbcc.labels_)) > 1

    with pytest.raises(ValueError):
        model.hbcc(boundary_use_reachability=False)

    with pytest.raises(ValueError):
        model.hbcc(hop_type="metric")


def test_hbcc_branches(distance_matrix):
    model = KMSTDescent(metric="precomputed").fit(distance_matrix)
    hbcc = model.hbcc(min_cluster_size=5)
    with pytest.raises(ValueError):
        model.branch_detector(hbcc)
