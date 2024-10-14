import numba
import numpy as np

from pynndescent import NNDescent
from .nn_descent import initialize_out_graph, nn_descent
from ..graph import as_knn
from ..disjoint_set import ds_rank_create, ds_find
from ..k_mst.heap import simple_edge_heap_push, deheap_sort_edges
from ..k_mst.boruvka import _merge_components, _initialize_boruvka_from_knn


def parallel_boruvka(
    data,
    metric,
    metric_kwds,
    num_neighbors,
    min_samples,
    epsilon,
    min_descent_neighbors,
    nn_kwargs,
):
    """Performs adapted parallel Boruvka's algorithm to find k-MST of a dataset."""
    # Initialize nearest neighbours
    index = NNDescent(
        data,
        metric=metric,
        metric_kwds=metric_kwds,
        n_neighbors=max(num_neighbors, min_descent_neighbors),
        **nn_kwargs,
    )
    neighbors, distances = index._neighbor_graph

    # Enter numba.njit context.
    return _internal_boruvka(
        index._raw_data,
        index._distance_func,
        neighbors,
        distances,
        num_neighbors,
        min_samples,
        epsilon,
        index.n_neighbors,
        min(60, index.n_neighbors),
        3 * index.n_iters,
        index.delta,
        index.rng_state
    )


@numba.njit()
def _internal_boruvka(
    data,
    dist,
    neighbors,
    distances,
    num_neighbors,
    min_samples,
    epsilon,
    descent_neighbors,
    max_candidates,
    n_iters,
    delta,
    rng_state
):
    """Runs the boruvka algorithm in numba.njit context."""
    # Allocate disjoint set and point components
    components_disjoint_set = ds_rank_create(data.shape[0])
    point_components = np.empty(data.shape[0], dtype=np.int32)
    core_distances = distances.T[min_samples]

    # Initialze Descent and output graph
    prev_neighbors, prev_distances = neighbors, distances
    graph = _initialize_boruvka_from_knn(
        neighbors,
        distances,
        core_distances,
        components_disjoint_set,
        num_neighbors,
        min_samples,
        epsilon,
    )

    # Iterate until all components are merged
    while True:
        # Update connected component state
        grouped_uniques, remapped_components = _update_component_state(
            components_disjoint_set, point_components
        )
        if len(grouped_uniques) <= 1:
            break

        # Update Descent graph state
        out_graph = initialize_out_graph(
            data,
            dist,
            (prev_neighbors, prev_distances),
            remapped_components,
            grouped_uniques,
            rng_state,
        )

        # Find nearest neighbours in other components
        nn_descent(
            data,
            dist,
            (neighbors, distances),
            out_graph,
            remapped_components,
            max_candidates,
            n_iters,
            delta,
            rng_state,
        )
        prev_neighbors, prev_distances = out_graph[:2]

        # Aggregate to shortest edges per component
        # Add edges and update disjoint set
        _merge_components(
            graph,
            components_disjoint_set,
            _select_candidates(
                out_graph[0],
                out_graph[1],
                remapped_components,
                len(grouped_uniques),
            ),
            num_neighbors,
            epsilon,
        )
    return as_knn(graph)


@numba.njit(parallel=True, cache=True)
def _update_component_state(disjoint_set, point_components):
    """Updates the point_components array."""
    # List component IDs for each point
    for i in numba.prange(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))

    # Create a list with indices of each unique value
    # grouped_idx[remapped[6]] contains data point indices belonging to component 6!
    sort_idx = np.argsort(point_components)
    sorted = point_components[sort_idx]
    unq_first = np.empty(sorted.shape, dtype=np.bool_)
    unq_first[0] = True
    unq_first[1:] = sorted[1:] != sorted[:-1]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    grouped_idx = []
    start = 0
    for count in unq_count:
        grouped_idx.append(sort_idx[start : start + count])
        start += count
    grouped_idx.append(sort_idx[start:])

    # Remap component IDs to index into grouped_idx
    remapped = np.empty_like(point_components)
    for i in numba.prange(len(grouped_idx)):
        remapped[grouped_idx[i]] = i

    return grouped_idx, remapped


@numba.njit(locals={"i": numba.types.int32}, cache=True)
def _select_candidates(
    candidate_neighbors,
    candidate_neighbor_distances,
    remapped_point_components,
    n_components,
):
    """Aggregates best candidate edges to the component and sorts them by distance."""
    shape = (n_components, candidate_neighbors.shape[1])
    component_heap_p = np.full(shape, np.inf, dtype=np.float32)
    component_heap_s = np.full(shape, -1, dtype=np.int32)
    component_heap_t = np.full(shape, -1, dtype=np.int32)

    for i, (neighbors, distances) in enumerate(
        zip(candidate_neighbors, candidate_neighbor_distances)
    ):
        from_component = np.int32(remapped_point_components[i])
        for d, n in zip(distances, neighbors):
            if d >= component_heap_p[from_component, 0]:
                break
            simple_edge_heap_push(
                component_heap_p[from_component, :],
                component_heap_s[from_component, :],
                component_heap_t[from_component, :],
                d,
                i,
                n,
            )
    return deheap_sort_edges(component_heap_p, component_heap_s, component_heap_t)
