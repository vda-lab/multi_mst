import numba
import numpy as np

from pynndescent import NNDescent
from .nn_descent import initialize_out_graph, nn_descent
from ..graph import as_knn
from ..disjoint_set import ds_rank_create, ds_find
from ..k_mst_descent.boruvka import _select_candidates
from ..k_mst.boruvka import (
    _boruvka_tree_query, _merge_components, _initialize_boruvka_from_knn
)


def parallel_boruvka(
    data,
    tree,
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
        metric="euclidean",
        metric_kwds={},
        n_neighbors=max(num_neighbors, min_descent_neighbors),
        **nn_kwargs,
    )
    neighbors, distances = index._neighbor_graph
    
    # Enter numba.njit context.
    return _internal_boruvka(
        index._raw_data,
        index._distance_func,
        tree,
        neighbors,
        distances,
        num_neighbors,
        min_samples,
        epsilon,
        min(60, index.n_neighbors),
        3 * index.n_iters,
        index.delta,
        index.rng_state,
    )


@numba.njit()
def _internal_boruvka(
    data,
    dist,
    tree,
    neighbors,
    distances,
    num_neighbors,
    min_samples,
    epsilon,
    max_candidates,
    n_iters,
    delta,
    rng_state,
):
    """Runs the boruvka algorithm in numba.njit context."""
    # Allocate disjoint set and point components
    components_disjoint_set = ds_rank_create(data.shape[0])
    point_components = np.empty(data.shape[0], dtype=np.int32)
    node_components = np.full(tree.node_data.shape[0], -1)
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
    trace = []
    while True:
        # Update connected component state
        grouped_uniques, remapped_components = _update_component_state(
            tree, components_disjoint_set, node_components, point_components
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
        
        # Find ground truth neighbours in other components
        true_distances, true_neighbours = _boruvka_tree_query(
            tree, node_components, point_components, core_distances, num_neighbors
        )
        true_selected = _select_candidates(
            true_neighbours,
            true_distances,
            remapped_components,
            len(grouped_uniques),
        )

        # Find approx nearest neighbours in other components
        recall_trace, dist_frac_trace, change_trace = nn_descent(
            data,
            dist,
            (neighbors, distances),
            out_graph,
            (true_distances, true_neighbours),
            true_selected,
            remapped_components,
            max_candidates,
            n_iters,
            delta,
            rng_state,
        )
        prev_neighbors, prev_distances = out_graph[:2]

        # Aggregate to shortest edges per component
        pred_selected = _select_candidates(
            out_graph[0][:, :num_neighbors],
            out_graph[1][:, :num_neighbors],
            remapped_components,
            len(grouped_uniques),
        )
        # Compute recall of selected edges
        boruvka_recall = _compute_recall(true_selected, pred_selected)
        trace.append(
            (
                len(grouped_uniques),
                boruvka_recall,
                recall_trace,
                dist_frac_trace,
                change_trace,
            )
        )

        # Add edges and update disjoint set
        _merge_components(
            graph,
            components_disjoint_set,
            pred_selected,
            num_neighbors,
            epsilon,
        )
    return as_knn(graph), trace


@numba.njit(parallel=True, cache=True)
def _update_component_state(tree, disjoint_set, node_components, point_components):
    """Updates the point_components array."""
    # List component IDs for each point
    for i in numba.prange(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))

    # Update KDTree state
    for i in range(tree.node_data.shape[0] - 1, -1, -1):
        node_info = tree.node_data[i]

        # Case 1: If the node is a leaf we need to check that every point in the
        #    node is of the same component
        if node_info.is_leaf:
            candidate_component = point_components[tree.idx_array[node_info.idx_start]]
            for j in range(node_info.idx_start + 1, node_info.idx_end):
                idx = tree.idx_array[j]
                if point_components[idx] != candidate_component:
                    break
            else:
                node_components[i] = candidate_component

        # Case 2: If the node is not a leaf we only need to check that both
        #    child nodes are in the same component
        else:
            left = 2 * i + 1
            right = left + 1

            if node_components[left] == node_components[right]:
                node_components[i] = node_components[left]

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


@numba.njit()
def _compute_recall(true_selected, pred_selected):
    num_points = true_selected[0].shape[0]
    num_neighbours = true_selected[0].shape[1]

    # Compute intersection (sparse multiplication)
    true_positives = 0
    for pt in range(num_points):
        i1 = 0
        i2 = 0
        while max(i1, i2) < num_neighbours:
            t_d = true_selected[0][pt, i1]
            t_s = true_selected[1][pt, i1]
            t_t = true_selected[2][pt, i1]
            p_d = pred_selected[0][pt, i1]
            p_s = pred_selected[1][pt, i1]
            p_t = pred_selected[2][pt, i1]
            if min(t_s, t_t, p_s, p_d) < 0:
                break
            if t_s == p_s and t_t == p_t:
                true_positives += 1
                i1 += 1
                i2 += 1
            elif t_d < p_d:
                i1 += 1
            else:
                i2 += 1

    return true_positives / (true_selected[0].shape[0] * num_neighbours)
