import numba
import numpy as np

from scipy.sparse import csr_array
from fast_hdbscan.core_graph import (
    CoreGraph,
    apply_lens,
    minimum_spanning_tree,
    clusters_from_spanning_tree,
)


def lensed_spanning_tree_from_graph(graph, lens):
    g = apply_lens(
        CoreGraph(np.empty_like(graph.data), graph.data, graph.indices, graph.indptr),
        lens,
    )
    return minimum_spanning_tree(g)[-1], g


@numba.njit(cache=True)
def csr_to_neighbor_list(data, indices, indptr):
    n = indptr.shape[0] - 1
    k = np.diff(indptr).max()
    mst_indices = np.full((n, k), -1, dtype=np.int32)
    mst_distances = np.full((n, k), np.inf, dtype=np.float32)

    for i in numba.prange(n):
        start = indptr[i]
        end = indptr[i + 1]
        row_indices = indices[start:end]
        row_data = data[start:end]
        order = np.argsort(row_data)
        mst_indices[i, : (end - start)] = row_indices[order]
        mst_distances[i, : (end - start)] = row_data[order]

    return mst_indices, mst_distances


@numba.njit(parallel=True, cache=True)
def edgelist_mutual_reachability(edges, core_distances):
    for idx in numba.prange(edges.shape[0]):
        parent = np.int32(edges[idx, 0])
        child = np.int32(edges[idx, 1])
        edges[idx, 2] = max(
            core_distances[parent], core_distances[child], edges[idx, 2]
        )
    return edges


def csr_mutual_reachability(graph, core_distances):
    _csr_mutual_reachability(graph.data, graph.indices, graph.indptr, core_distances)
    return graph


@numba.njit(parallel=True, cache=True)
def _csr_mutual_reachability(distances, indices, indptr, core_distances):
    for parent in numba.prange(core_distances.shape[0]):
        # get the row
        start = indptr[parent]
        end = indptr[parent + 1]

        # fill the row
        row_indices = indices[start:end]
        row_distances = distances[start:end]
        for idx, (child, dist) in enumerate(zip(row_indices, row_distances)):
            row_distances[idx] = max(
                core_distances[parent], core_distances[child], dist
            )

        # sort the row
        order = np.argsort(row_distances)
        indices[start:end] = row_indices[order]
        distances[start:end] = row_distances[order]


def make_csr_graph(mst_edges, k_edges, num_points):
    return csr_array(
        _to_csr_graph(mst_edges, k_edges, num_points),
        shape=(num_points, num_points),
    )


@numba.njit(cache=True)
def _to_csr_graph(mst_edges, k_edges, num_points):
    graph = [
        {np.int32(0): np.float32(0.0) for _ in range(0)}
        for point in range(num_points)
    ]
    _add_edgelist(graph, mst_edges)
    _add_edgelist(graph, k_edges)

    return _flatten_to_sorted_csr(graph, num_points)


@numba.njit(cache=True)
def _add_edgelist(graph, edges):
    for parent, child, distance in edges:
        parent = np.int32(parent)
        child = np.int32(child)
        children = graph[parent]
        if child in children:
            continue
        children[child] = np.float32(distance)


@numba.njit(parallel=True, cache=True)
def _flatten_to_sorted_csr(graph, num_points):
    indptr = np.empty(num_points + 1, dtype=np.int32)
    indptr[0] = 0
    for i, children in enumerate(graph):
        indptr[i + 1] = indptr[i] + len(children)

    distances = np.empty(indptr[-1], dtype=np.float32)
    indices = np.empty(indptr[-1], dtype=np.int32)
    for point in numba.prange(num_points):
        # get the row
        start = indptr[point]
        end = indptr[point + 1]

        # fill the row
        children = graph[point]
        row_indices = indices[start:end]
        row_distances = distances[start:end]
        for j, (child, distance) in enumerate(children.items()):
            row_indices[j] = child
            row_distances[j] = distance

        # sort the row
        order = np.argsort(row_distances)
        indices[start:end] = row_indices[order]
        distances[start:end] = row_distances[order]

    # Return as named csr tuple
    return (distances, indices, indptr)
