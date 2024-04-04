import numba
import numpy as np

from .heap import simple_edge_heap_push, deheap_sort_edges
from ..graph import init_graph, add_edge, as_knn
from ..disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from ..boruvka import update_component_vectors
from ..kdtree import (
    rdist,
    parallel_tree_query,
    point_to_node_lower_bound_rdist,
    simple_heap_push,
    deheap_sort,
)


@numba.njit(locals={"i": numba.types.int32})
def select_candidates(
    candidate_neighbors,
    candidate_neighbor_distances,
    point_components,
    unique_components,
):
    """
    Aggregates best candidate edges to the component and sorts them by distance.
    """
    n = unique_components.shape[0]
    k = candidate_neighbors.shape[1]
    component_idx_map = {
        np.int64(c): np.int32(i) for i, c in enumerate(unique_components)
    }
    component_heap_p = np.full((n, k), np.inf, dtype=np.float32)
    component_heap_s = np.full((n, k), -1, dtype=np.int32)
    component_heap_t = np.full((n, k), -1, dtype=np.int32)

    for i, (neighbors, distances) in enumerate(
        zip(candidate_neighbors, candidate_neighbor_distances)
    ):
        from_component = np.int32(point_components[i])
        component_idx = component_idx_map[from_component]
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


@numba.njit()
def merge_components(results, disjoint_set, component_edges, num_neighbors, epsilon):
    """
    Updates the disjoint set with selected candidate edges.
    """
    for distances, sources, targets in zip(*component_edges):
        from_component = ds_find(disjoint_set, sources[0])
        to_component = ds_find(disjoint_set, targets[0])
        if from_component != to_component:
            ds_union_by_rank(disjoint_set, from_component, to_component)
        add_edge(
            results,
            np.int32(sources[0]),
            np.int32(targets[0]),
            np.float32(distances[0]),
        )

        for i in range(1, num_neighbors):
            if targets[i] < 0 or distances[i] > (epsilon * distances[0]):
                break
            add_edge(
                results,
                np.int32(sources[i]),
                np.int32(targets[i]),
                np.float32(distances[i]),
            )


@numba.njit()
def component_aware_query_recursion(
    tree,
    node,
    point,
    heap_p,
    heap_i,
    current_core_distance,
    core_distances,
    current_component,
    node_components,
    point_components,
    dist_lower_bound,
    component_nearest_neighbor_dist,
):
    """
    Traverses a KD-tree recursively to find each point's nearest neighbor in
    another component.
    """
    node_info = tree.node_data[node]

    # ------------------------------------------------------------
    # Case 1a: query point is outside node radius: trim node from the query
    if dist_lower_bound > heap_p[0]:
        return

    # ------------------------------------------------------------
    # Case 1b: we can't improve on the best distance for this component trim
    #         node / point from the query
    elif (
        dist_lower_bound > component_nearest_neighbor_dist[0]
        or current_core_distance > component_nearest_neighbor_dist[0]
    ):
        return

    # ------------------------------------------------------------
    # Case 1c: node contains only points in same component as query trim node
    #         from the query
    elif node_components[node] == current_component:
        return

    # ------------------------------------------------------------
    # Case 2: this is a leaf node. Update set of nearby points
    elif node_info.is_leaf:
        for i in range(node_info.idx_start, node_info.idx_end):
            idx = tree.idx_array[i]
            if (
                point_components[idx] != current_component
                and core_distances[idx] < component_nearest_neighbor_dist[0]
            ):
                d = max(
                    rdist(point, tree.data[idx]),
                    current_core_distance,
                    core_distances[idx],
                )
                if d < heap_p[0]:
                    simple_heap_push(heap_p, heap_i, d, idx)
                    if heap_p[0] < component_nearest_neighbor_dist[0]:
                        component_nearest_neighbor_dist[0] = heap_p[0]

    # ------------------------------------------------------------
    # Case 3: Node is not a leaf. Recursively query subnodes starting with the
    #         closest
    else:
        left = 2 * node + 1
        right = left + 1
        dist_lower_bound_left = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, left], tree.node_bounds[1, left], point
        )
        dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, right], tree.node_bounds[1, right], point
        )

        # recursively query subnodes
        if dist_lower_bound_left <= dist_lower_bound_right:
            component_aware_query_recursion(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist,
            )
            component_aware_query_recursion(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist,
            )
        else:
            component_aware_query_recursion(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist,
            )
            component_aware_query_recursion(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist,
            )

    return


@numba.njit(parallel=True)
def boruvka_tree_query(tree, node_components, point_components, core_distances, k):
    """
    Finds the k (mutual reachability) closest neighbor in another component for
    each data point.
    """
    candidate_distances = np.full((tree.data.shape[0], k), np.inf, dtype=np.float32)
    candidate_indices = np.full((tree.data.shape[0], k), -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(
        tree.data.shape[0], np.inf, dtype=np.float32
    )

    data = np.asarray(tree.data.astype(np.float32))
    for i in numba.prange(tree.data.shape[0]):
        distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i]
        )
        heap_p, heap_i = candidate_distances[i], candidate_indices[i]
        component_aware_query_recursion(
            tree,
            0,
            data[i],
            heap_p,
            heap_i,
            core_distances[i],
            core_distances,
            point_components[i],
            node_components,
            point_components,
            distance_lower_bound,
            component_nearest_neighbor_dist[
                point_components[i] : point_components[i] + 1
            ],
        )

    return deheap_sort(candidate_distances, candidate_indices)


@numba.njit(locals={"i": numba.types.int32})
def initialize_boruvka_from_knn(
    knn_indices, knn_distances, core_distances, disjoint_set, min_samples, epsilon
):
    """
    Initializes Boruvka's algorithm from k-nearest neighbors indices.
    """
    graph = init_graph(knn_indices.shape[0])
    for i in range(knn_indices.shape[0]):
        # Add min_samples closest neighbours, guaranteed to be within core distance
        for j in range(1, min_samples + 1):
            k = np.int32(knn_indices[i, j])
            if core_distances[i] < core_distances[k]:
                continue
            from_component = ds_find(disjoint_set, np.int32(i))
            to_component = ds_find(disjoint_set, np.int32(k))
            if from_component != to_component:
                ds_union_by_rank(disjoint_set, from_component, to_component)
            add_edge(graph, i, k, np.float32(core_distances[i]))

        # Add additional neighbours if they are within epsilon * core distance
        for j in range(min_samples + 1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            mutual_core = max(core_distances[i], core_distances[k])
            d = max(knn_distances[i, j], mutual_core)
            if d >= epsilon * mutual_core:
                break
            add_edge(graph, i, k, np.float32(d))
    return graph


@numba.njit(parallel=True)
def parallel_boruvka(tree, num_neighbors, min_samples, epsilon):
    """
    Perform adapted parallel Boruvka's algorithm to find k-MST of a dataset.
    """
    components_disjoint_set = ds_rank_create(tree.data.shape[0])
    point_components = np.arange(tree.data.shape[0])
    node_components = np.full(tree.node_data.shape[0], -1)

    # Initialize from min_samples nearest neighbours
    distances, neighbors = parallel_tree_query(
        tree, tree.data, k=num_neighbors + 1, output_rdist=True
    )
    core_distances = distances.T[min_samples]
    graph = initialize_boruvka_from_knn(
        neighbors,
        distances,
        core_distances,
        components_disjoint_set,
        min_samples,
        epsilon,
    )
    update_component_vectors(
        tree, components_disjoint_set, node_components, point_components
    )

    unique_components = np.unique(point_components)
    n_components = unique_components.shape[0]
    while n_components > 1:
        candidate_distances, candidate_indices = boruvka_tree_query(
            tree, node_components, point_components, core_distances, num_neighbors
        )
        merge_components(
            graph,
            components_disjoint_set,
            select_candidates(
                candidate_indices,
                candidate_distances,
                point_components,
                unique_components,
            ),
            num_neighbors,
            epsilon,
        )
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components
        )

        unique_components = np.unique(point_components)
        n_components = unique_components.shape[0]

    return as_knn(graph)
