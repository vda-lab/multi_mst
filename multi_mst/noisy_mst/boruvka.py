import numba
import numpy as np

from ..graph import init_graph, add_edge, as_knn
from ..disjoint_set import ds_rank_create, ds_find, ds_union_by_rank, ds_rank_copy
from ..boruvka import update_component_vectors
from ..kdtree import point_to_node_lower_bound_rdist, parallel_tree_query


@numba.njit(parallel=True)
def parallel_boruvka(tree, num_trees, noise_fraction, min_samples):
    """
    Perform adapted parallel Boruvka's algorithm to find a union of k noisy
    MSTs.
    """
    graph = init_graph(tree.data.shape[0])
    initial_disjoint_set = ds_rank_create(tree.data.shape[0])

    distances, neighbors = parallel_tree_query(
        tree, tree.data, k=min_samples + 1, output_rdist=True
    )
    core_distances = distances.T[min_samples]
    _initialize_boruvka_from_knn(graph, neighbors, core_distances, initial_disjoint_set)

    for _ in range(num_trees):
        components_disjoint_set = ds_rank_copy(initial_disjoint_set)
        point_components = np.arange(tree.data.shape[0])
        node_components = np.full(tree.node_data.shape[0], -1)
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components
        )

        unique_components = np.unique(point_components)
        n_components = unique_components.shape[0]
        while n_components > 1:
            candidate_distances, candidate_indices = _boruvka_tree_query(
                tree, node_components, point_components, core_distances, noise_fraction
            )
            _merge_components(
                graph,
                components_disjoint_set,
                candidate_indices,
                candidate_distances,
                point_components,
            )
            update_component_vectors(
                tree, components_disjoint_set, node_components, point_components
            )

            unique_components = np.unique(point_components)
            n_components = unique_components.shape[0]
    return as_knn(graph)


@numba.njit(locals={"i": numba.types.int32})
def _initialize_boruvka_from_knn(graph, knn_indices, core_distances, disjoint_set):
    """Initializes Boruvka's algorithm from k-nearest neighbors indices."""
    component_sources = np.full(knn_indices.shape[0], -1, dtype=np.int32)
    component_targets = np.full(knn_indices.shape[0], -1, dtype=np.int32)
    component_dists = np.full(knn_indices.shape[0], np.inf, dtype=np.float32)

    for i in numba.prange(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            if core_distances[i] >= core_distances[k]:
                component_sources[i] = i
                component_targets[i] = k
                component_dists[i] = core_distances[i]
                break

    # Add the best edges to the edge set and merge the relevant components
    for i, j, d in zip(component_sources, component_targets, component_dists):
        if i < 0:
            continue
        from_component = ds_find(disjoint_set, i)
        to_component = ds_find(disjoint_set, j)
        if from_component != to_component:
            add_edge(graph, i, j, d)
            ds_union_by_rank(disjoint_set, from_component, to_component)


@numba.njit(locals={"i": numba.types.int64})
def _merge_components(
    graph,
    disjoint_set,
    candidate_neighbors,
    candidate_neighbor_distances,
    point_components,
):
    """Adds the best edge from each component and updates disjoint set."""
    component_edges = {
        np.int32(0): (np.int32(0), np.int32(1), np.float32(0.0)) for i in range(0)
    }

    # Find the best edges from each component
    for i in range(candidate_neighbors.shape[0]):
        from_component = np.int32(point_components[i])
        if from_component in component_edges:
            if candidate_neighbor_distances[i] < component_edges[from_component][2]:
                component_edges[from_component] = (
                    np.int32(i),
                    np.int32(candidate_neighbors[i]),
                    candidate_neighbor_distances[i],
                )
        else:
            component_edges[from_component] = (
                np.int32(i),
                np.int32(candidate_neighbors[i]),
                candidate_neighbor_distances[i],
            )

    # Add the best edges to the edge set and merge the relevant components
    for edge in component_edges.values():
        from_component = ds_find(disjoint_set, edge[0])
        to_component = ds_find(disjoint_set, edge[1])
        if from_component != to_component:
            add_edge(graph, edge[0], edge[1], edge[2])
            ds_union_by_rank(disjoint_set, from_component, to_component)


@numba.njit(parallel=True)
def _boruvka_tree_query(
    tree, node_components, point_components, core_distances, noise_fraction
):
    """
    Finds the k (mutual reachability) closest neighbor in another component
    for each data point.
    """
    candidate_distances = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    candidate_indices = np.full(tree.data.shape[0], -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(
        tree.data.shape[0], np.inf, dtype=np.float32
    )

    data = np.asarray(tree.data.astype(np.float32))
    for i in numba.prange(tree.data.shape[0]):
        distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i]
        )
        heap_p, heap_i = candidate_distances[i : i + 1], candidate_indices[i : i + 1]
        _component_aware_query_recursion(
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
            noise_fraction,
        )

    return candidate_distances, candidate_indices


@numba.njit()
def _component_aware_query_recursion(
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
    noise_fraction,
):
    """
    Traverses a KD-tree recursively to find each point's nearest neighbor
    in another component.
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
                mutual_core = max(current_core_distance, core_distances[idx])
                d = _noisy_rdist(point, tree.data[idx], noise_fraction * mutual_core)
                d = max(d, mutual_core)
                if d < heap_p[0]:
                    heap_p[0] = d
                    heap_i[0] = idx
                    if d < component_nearest_neighbor_dist[0]:
                        component_nearest_neighbor_dist[0] = d

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
            _component_aware_query_recursion(
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
                noise_fraction,
            )
            _component_aware_query_recursion(
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
                noise_fraction,
            )
        else:
            _component_aware_query_recursion(
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
                noise_fraction,
            )
            _component_aware_query_recursion(
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
                noise_fraction,
            )

    return


@numba.njit(
    [
        "f4(f4[::1],f4[::1],f4)",
        "f8(f8[::1],f8[::1],f4)",
        "f8(f4[::1],f8[::1],f4)",
    ],
    fastmath=True,
    locals={
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def _noisy_rdist(x, y, std_dev):
    """Computes a noisy squared Euclidean distance between two points."""
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff
    result += np.random.normal(0.0, std_dev, 1)[0]
    return result