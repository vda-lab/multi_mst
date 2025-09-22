import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


@pytest.fixture(scope="session")
def X():
    blobs, y = make_blobs(
        n_samples=50,
        centers=[(-0.75, 2.25), (2.0, -0.5)],
        cluster_std=0.2,
        random_state=3,
    )
    np.random.seed(5)
    noise = np.random.uniform(-1.0, 3.0, (50, 2))
    return StandardScaler().fit_transform(np.vstack((blobs, noise)))


@pytest.fixture(scope="session")
def X_missing(X):
    X_missing_data = X.copy()
    X_missing_data[0] = [np.nan, 1]
    X_missing_data[6] = [np.nan, np.nan]
    return X_missing_data


@pytest.fixture(scope="session")
def clean_indices(X_missing):
    return [i for i in range(X_missing.shape[0]) if not np.isnan(X_missing[i]).any()]


@pytest.fixture(scope="session")
def distance_matrix(X):
    return squareform(pdist(X))
