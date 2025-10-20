import os
import warnings
import numpy as np
from sklearn.datasets import load_iris as sk_load_iris
from sklearn.datasets import load_diabetes as sk_load_diabetes
from sklearn.datasets import load_wine as sk_load_wine
from sklearn.datasets import fetch_openml

def load_iris():
    X, _ = sk_load_iris(return_X_y=True)
    return X


def load_diabetes():
    X, _ = sk_load_diabetes(return_X_y=True)
    return X


def load_wine():
    X, _ = sk_load_wine(return_X_y=True)
    return X


def load_mnist():
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True)
    return X


def load_fashion_mnist():
    X, _ = fetch_openml("Fashion-MNIST", version=1, return_X_y=True)
    return X


def local_data_loader(folder):
    """Load data from a specified folder in the data directory.

    See `data/[data-set]/README.md` for processing details.
    """
    X_path = f"data/{folder}/generated/X.npy"
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Data not found in folder: {folder}")

    def load_data():
        X = np.load(X_path)
        return X

    return load_data


def get_datasets():
    data_configs = dict(
        iris=load_iris,
        diabetes=load_diabetes,
        wine=load_wine,
        mnist=load_mnist,
        fashion_mnist=load_fashion_mnist,
    )

    for folder in os.listdir("data"):
        if folder == "generated":
            continue
        if not os.path.isdir(os.path.join("data", folder)):
            continue

        try:
            loader = local_data_loader(folder)
            data_configs[folder] = loader
        except FileNotFoundError as e:
            warnings.warn(
                f"Skipping {folder}. See "
                f"`data/{folder}/README.md` for download instructions."
            )

    return data_configs
