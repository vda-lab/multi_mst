"""Tests for MultiMST"""
import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_raises
from scipy.sparse.csgraph import connected_components

from umap import UMAP
from multi_mst.noisy_mst import noisyMST, NoisyMST


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

X_missing_data = X.copy()
X_missing_data[0] = [np.nan, 1]
X_missing_data[6] = [np.nan, np.nan]


def test_badargs():
    """Tests parameter validation."""
    assert_raises(ValueError, noisyMST, X, num_trees=1.0)
    assert_raises(ValueError, noisyMST, X, num_trees=-1)
    assert_raises(ValueError, noisyMST, X, noise_fraction=1)
    assert_raises(ValueError, noisyMST, X, noise_fraction=-0.4)
    assert_raises(ValueError, noisyMST, X, min_samples=1.0)
    assert_raises(ValueError, noisyMST, X, min_samples=0)
    assert_raises(ValueError, noisyMST, X, min_samples=-1)


def test_defaults():
    """Tests with default parameters."""
    p = NoisyMST()
    
    embedding = p.fit_transform(X)
    assert embedding.shape[0] == X.shape[0]
    assert embedding.shape[1] == 2 # Default num_components
    assert np.issubdtype(embedding.dtype, np.floating)

    assert p.mst_indices_.shape[0] == X.shape[0]
    assert p.mst_indices_.shape[1] >= 1
    assert np.issubdtype(p.mst_indices_.dtype, np.integer)
    assert p.mst_distances_.shape[0] == X.shape[0]
    assert p.mst_distances_.shape[1] >= 1
    assert np.issubdtype(p.mst_distances_.dtype, np.floating)
    assert p.graph_.shape[0] == X.shape[0]
    assert p.graph_.shape[1] == X.shape[0]
    assert p.embedding_.shape[0] == X.shape[0]
    assert p.embedding_.shape[1] == 2 # Default num_components
    assert np.issubdtype(p.embedding_.dtype, np.floating)
    assert isinstance(p._umap, UMAP)
    assert connected_components(p._umap.graph_, directed=False, return_labels=False) == 1


def test_with_missing_data():
    """Tests with nan data."""
    model = NoisyMST().fit(X_missing_data)
    
    assert np.all(model.mst_indices_[0, :] == -1)
    assert np.isinf(model.mst_distances_[0, :]).all()
    assert np.isnan(model.embedding_[0, :]).all()
    assert np.all(model.graph_.row != 0) & np.all(model.graph_.col != 0)

    assert np.all(model.mst_indices_[6, :] == -1)
    assert np.isinf(model.mst_distances_[6, :]).all()
    assert np.isnan(model.embedding_[6, :]).all()
    assert np.all(model.graph_.row != 6) & np.all(model.graph_.col != 6)
    

def test_with_missing_data_graph_mode():
    """Tests with nan data."""
    model = NoisyMST(transform_mode='graph').fit(X_missing_data)
    
    assert np.all(model.mst_indices_[0, :] == -1)
    assert np.isinf(model.mst_distances_[0, :]).all()
    assert np.all(model.graph_.row != 0) & np.all(model.graph_.col != 0)

    assert np.all(model.mst_indices_[6, :] == -1)
    assert np.isinf(model.mst_distances_[6, :]).all()
    assert np.all(model.graph_.row != 6) & np.all(model.graph_.col != 6)


def test_graph_mode():
    """Tests with transform_mode='graph'."""
    p = NoisyMST(transform_mode='graph')
    
    embedding = p.fit_transform(X)
    assert embedding is None
    assert p.embedding_ is None
    

def test_min_samples():
    """Tests with higher min_samples."""
    p = NoisyMST(min_samples=3, transform_mode='graph').fit(X)
    
    assert p.mst_indices_.shape[1] >= 1
    assert p.mst_distances_.shape[1] >= 1
    assert connected_components(p._umap.graph_, directed=False, return_labels=False) == 1


def test_num_trees():
    """Tests with lower num_trees."""
    p = NoisyMST(num_trees=1, transform_mode='graph').fit(X)
    
    assert p.mst_indices_.shape[1] >= 1
    assert p.mst_distances_.shape[1] >= 1
    assert connected_components(p._umap.graph_, directed=False, return_labels=False) == 1


def test_noise_fraction():
    """Tests with higher noise_fraction."""
    p = NoisyMST(noise_fraction=0.5, transform_mode='graph').fit(X)
      
    assert p.mst_indices_.shape[1] >= 1
    assert p.mst_distances_.shape[1] >= 1
    assert connected_components(p._umap.graph_, directed=False, return_labels=False) == 1


def test_num_components():
    """Tests with higher num_components."""
    p = NoisyMST(n_components=3)
    
    embedding = p.fit_transform(X)
    assert embedding.shape[0] == X.shape[0]
    assert embedding.shape[1] == 3
    assert np.issubdtype(embedding.dtype, np.floating)

    assert p.embedding_.shape[0] == X.shape[0]
    assert p.embedding_.shape[1] == 3
    assert np.issubdtype(p.embedding_.dtype, np.floating)