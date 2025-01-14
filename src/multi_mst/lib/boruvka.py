import numba
import numpy as np

from .heap import simple_edge_heap_push, deheap_sort_edges
from fast_hdbscan.boruvka import update_point_components
from fast_hdbscan.disjoint_set import (
    RankDisjointSet,
    ds_find,
    ds_rank_create,
    ds_union_by_rank,
)


def multi_boruvka(index, epsilon):
    """Keeps the k-shortest edges for all components in each Boruvka iteration."""
    components_disjoint_set = ds_rank_create(index.num_points)
    point_components = np.arange(index.num_points, dtype=np.int32)
    n_components = point_components.shape[0]

    # Initialize from nearest neighbors
    k_edges = []
    mst_edges = []
    distances, neighbors = index.neighbors()
    new_mst, new_k = _initialize_boruvka_from_knn(
        neighbors[:, 1:],
        distances[:, 1:],
        index.core_distances,
        components_disjoint_set,
        index.num_neighbors,
        index.min_samples,
        epsilon,
    )
    while True:
        k_edges.append(new_k)
        mst_edges.append(new_mst)
        update_point_components(components_disjoint_set, point_components)
        n_components -= new_mst.shape[0]
        if n_components <= 1:
            break

        candidate_distances, candidate_indices = index.query(point_components)

        new_mst, new_k = _merge_components(
            components_disjoint_set,
            _select_candidates(
                candidate_indices, candidate_distances, point_components
            ),
            index.num_neighbors,
            epsilon,
        )

    k_edges = np.vstack(k_edges)
    mst_edges = np.vstack(mst_edges)
    distances = index.correction(distances)
    k_edges[:, 2] = index.correction(k_edges[:, 2])
    mst_edges[:, 2] = index.correction(mst_edges[:, 2])
    return mst_edges, k_edges, neighbors[:, 1:], distances[:, 1:]


def multi_boruvka_recall(true_index, recall_index, epsilon):
    """Runs multi_boruvka but also tracks recall compared to true_index."""
    components_disjoint_set = ds_rank_create(true_index.num_points)
    point_components = np.arange(true_index.num_points, dtype=np.int32)
    n_components = point_components.shape[0]

    trace = []
    k_edges = []
    mst_edges = []

    distances, neighbors = recall_index.neighbors()
    true_index.core_distances = recall_index.core_distances
    new_mst, new_k = _initialize_boruvka_from_knn(
        neighbors[:, 1:],
        distances[:, 1:],
        true_index.core_distances,
        components_disjoint_set,
        true_index.num_neighbors,
        true_index.min_samples,
        epsilon,
    )

    while True:
        k_edges.append(new_k)
        mst_edges.append(new_mst)
        n_components -= new_mst.shape[0]
        update_point_components(components_disjoint_set, point_components)
        if n_components <= 1:
            break

        true_distances, true_indices = true_index.query(point_components)
        true_selected = _select_candidates(
            true_indices, true_distances, point_components
        )
        candidate_distances, candidate_indices, traces = recall_index.query(
            point_components, (true_distances, true_indices), true_selected
        )
        candidate_selected = _select_candidates(
            candidate_indices, candidate_distances, point_components
        )
        boruvka_recall = _compute_recall(true_selected, candidate_selected)
        trace.append((n_components, boruvka_recall, *traces))

        new_mst, new_k = _merge_components(
            components_disjoint_set,
            candidate_selected,
            recall_index.num_neighbors,
            epsilon,
        )
        

    k_edges = np.vstack(k_edges)
    mst_edges = np.vstack(mst_edges)
    distances = recall_index.correction(distances)
    k_edges[:, 2] = recall_index.correction(k_edges[:, 2])
    mst_edges[:, 2] = recall_index.correction(mst_edges[:, 2])
    return mst_edges, k_edges, neighbors[:, 1:], distances[:, 1:], trace


def noisy_boruvka_union(noisy_index):
    """Computes multiple Boruvka MSTs with noisy distances."""
    k_edges = []
    mst_edges = []
    initial_disjoint_set = ds_rank_create(noisy_index.num_points)
    initial_point_components = np.arange(noisy_index.num_points, dtype=np.int32)

    distances, neighbors = noisy_index.neighbors()
    new_mst, _ = _initialize_boruvka_from_knn(
        neighbors[:, 1:],
        distances[:, 1:],
        noisy_index.core_distances,
        initial_disjoint_set,
        noisy_index.min_samples,
        noisy_index.min_samples,
        np.inf,
    )
    mst_edges.append(new_mst)
    update_point_components(initial_disjoint_set, initial_point_components)
    initial_n_components = noisy_index.num_points - mst_edges[0].shape[0]

    for i in range(noisy_index.num_neighbors):
        n_components = initial_n_components
        point_components = initial_point_components.copy()
        components_disjoint_set = ds_rank_copy(initial_disjoint_set)

        while n_components > 1:
            candidate_distances, candidate_indices = noisy_index.query(point_components)
            new_mst_edges, _ = _merge_components(
                components_disjoint_set,
                _select_candidates(
                    candidate_indices, candidate_distances, point_components
                ),
                1,
                np.inf,
            )
            if i == 0:
                mst_edges.append(new_mst_edges)
            else:
                k_edges.append(new_mst_edges)
            n_components -= new_mst_edges.shape[0]
            update_point_components(components_disjoint_set, point_components)
    
    if len(k_edges) == 0:
        k_edges = np.empty((0, 3))
    else:
        k_edges = np.vstack(k_edges)
    mst_edges = np.vstack(mst_edges)
    distances = noisy_index.correction(distances)
    k_edges[:, 2] = noisy_index.correction(k_edges[:, 2])
    mst_edges[:, 2] = noisy_index.correction(mst_edges[:, 2])
    return mst_edges, k_edges, neighbors[:, 1:], distances[:, 1:]


@numba.njit(locals={"i": numba.types.int32}, cache=True)
def _initialize_boruvka_from_knn(
    knn_indices,
    knn_distances,
    core_distances,
    disjoint_set,
    num_neighbors,
    min_samples,
    epsilon,
):
    """Initializes Boruvka's algorithm from k-nearest neighbors indices."""
    k_idx = 0
    mst_idx = 0
    num_components = knn_indices.shape[0]
    mst_result = np.empty((num_components, 3), dtype=np.float64)
    k_result = np.empty((num_components * num_neighbors, 3), dtype=np.float64)

    # Add nearest neighbor if they are not in the same component
    for i in range(num_components):
        k = knn_indices[i, 0]
        if core_distances[i] < core_distances[k]:
            continue
        edge = (np.float64(i), np.float64(k), np.float64(core_distances[i]))
        from_component = ds_find(disjoint_set, np.int32(i))
        to_component = ds_find(disjoint_set, np.int32(k))
        if from_component != to_component:
            ds_union_by_rank(disjoint_set, from_component, to_component)
            mst_result[mst_idx] = edge
            mst_idx += 1
        else:
            k_result[k_idx] = edge
            k_idx += 1

        # Add additional min_samples neighbors to k_edges
        for j in range(1, min_samples):
            k = np.int32(knn_indices[i, j])
            if core_distances[i] < core_distances[k]:
                continue
            k_result[k_idx] = (
                np.float64(i),
                np.float64(k),
                np.float64(core_distances[i]),
            )
            k_idx += 1

        # Add additional neighbors if they are within epsilon * core distance
        for j in range(min_samples + 1, num_neighbors):
            k = np.int32(knn_indices[i, j])
            mutual_core = max(core_distances[i], core_distances[k])
            d = max(knn_distances[i, j], mutual_core)
            if d >= epsilon * mutual_core:
                break

            k_result[k_idx] = (np.float64(i), np.float64(k), np.float64(d))
            k_idx += 1
    return mst_result[:mst_idx], k_result[:k_idx]


@numba.njit()
def _merge_components(disjoint_set, component_edges, num_neighbors, epsilon):
    """
    Updates the disjoint set with selected candidate edges.
    """
    k_idx = 0
    mst_idx = 0
    num_components = component_edges[0].shape[0]
    mst_result = np.empty((num_components, 3), dtype=np.float64)
    k_result = np.empty((num_components * num_neighbors, 3), dtype=np.float64)

    for distances, sources, targets in zip(*component_edges):
        from_component = ds_find(disjoint_set, sources[0])
        to_component = ds_find(disjoint_set, targets[0])
        edge = (
            np.float64(sources[0]),
            np.float64(targets[0]),
            np.float64(distances[0]),
        )
        if from_component != to_component:
            ds_union_by_rank(disjoint_set, from_component, to_component)
            mst_result[mst_idx] = edge
            mst_idx += 1
        else:
            k_result[k_idx] = edge
            k_idx += 1

        for i in range(1, num_neighbors):
            if targets[i] < 0 or distances[i] > (epsilon * distances[0]):
                break
            k_result[k_idx] = (
                np.float64(sources[i]),
                np.float64(targets[i]),
                np.float64(distances[i]),
            )
            k_idx += 1
    return mst_result[:mst_idx], k_result[:k_idx]


@numba.njit(locals={"i": numba.types.int32, "cnt": numba.types.int32})
def _select_candidates(
    candidate_neighbors, candidate_neighbor_distances, point_components
):
    """
    Aggregates best candidate edges to the component and sorts them by distance.
    """
    cnt = 0
    idx_map = {np.int32(0): np.int32(0) for _ in range(0)}
    for component in point_components:
        if component not in idx_map:
            idx_map[component] = cnt
            cnt += 1

    n = len(idx_map)
    k = candidate_neighbors.shape[1]
    component_heap_p = np.full((n, k), np.inf, dtype=np.float32)
    component_heap_s = np.full((n, k), -1, dtype=np.int32)
    component_heap_t = np.full((n, k), -1, dtype=np.int32)

    for i, (neighbors, distances) in enumerate(
        zip(candidate_neighbors, candidate_neighbor_distances)
    ):
        from_component = np.int32(point_components[i])
        component_idx = idx_map[from_component]
        for d, n in zip(distances, neighbors):
            if d >= component_heap_p[component_idx, 0]:
                break
            simple_edge_heap_push(
                component_heap_p[component_idx, :],
                component_heap_s[component_idx, :],
                component_heap_t[component_idx, :],
                d,
                i,
                n,
            )
    return deheap_sort_edges(component_heap_p, component_heap_s, component_heap_t)


@numba.njit(cache=True, fastmath=True)
def _compute_recall(true_selected, pred_selected):
    num_points = true_selected[0].shape[0]
    num_neighbors = true_selected[0].shape[1]

    # Compute intersection (sparse multiplication)
    true_positives = 0
    for pt in range(num_points):
        i1 = 0
        i2 = 0
        while max(i1, i2) < num_neighbors:
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

    return true_positives / (true_selected[0].shape[0] * num_neighbors)


@numba.njit()
def ds_rank_copy(disjoint_set):
    """Creates a copy of the disjoint set."""
    return RankDisjointSet(disjoint_set.parent.copy(), disjoint_set.rank.copy())
