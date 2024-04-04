import numba
import numpy as np


@numba.njit()
def init_graph(num_points):
    """Initializes a graph as a list of dictionaries [{target: distance,...},...]."""
    return [{np.int32(0): np.float32(0.0) for _ in range(0)} for _ in range(num_points)]


@numba.njit()
def add_edge(graph, source, target, weight):
    """
    Adds a directed edge to a list of dictionaries.
    graph = [{target: weight, ...}, ...]
             ^ index = source
    """
    graph[source][target] = weight


@numba.njit(
    locals={"i": numba.types.int32, "k": numba.types.int32, "n": numba.types.int32},
    parallel=True,
)
def as_knn(graph):
    """Convert graph format to knn_indices and knn_distances format."""
    k = 0
    for d in graph:
        k = max(k, len(d))

    mst_indices = np.full((len(graph), k), -1, dtype=np.int32)
    mst_distances = np.full((len(graph), k), np.inf, dtype=np.float32)

    for i in numba.prange(len(graph)):
        d = graph[i]
        indices = np.empty(len(d), dtype=np.int32)
        distances = np.empty(len(d), dtype=np.float32)
        for j, (k, v) in enumerate(d.items()):
            indices[j] = k
            distances[j] = np.sqrt(v)

        order = np.argsort(distances)
        for j, o in enumerate(order):
            mst_indices[i, j] = indices[o]
            mst_distances[i, j] = distances[o]
    return mst_indices, mst_distances
