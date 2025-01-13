import numba
import numpy as np

from .kdtree import (
    KDTree,
    KDTreeIndex,
    rdist,
    simple_heap_push,
    deheap_sort,
    update_node_components,
    point_to_node_lower_bound_rdist,
    _component_aware_stop_condition,
)


class NoisyKDTreeIndex(KDTreeIndex):
    def __init__(self, data, num_neighbors=5, min_samples=1, noise_fraction=0.1):
        super().__init__(data, num_neighbors, min_samples)
        self.noise_fraction = noise_fraction

    def neighbors(self):
        n_neighbors = self.num_neighbors
        self.num_neighbors = self.min_samples
        result = super().neighbors()
        self.num_neighbors = n_neighbors
        return result

    def query(self, point_components):
        update_node_components(self.tree, self.node_components, point_components)
        return boruvka_tree_query(
            self.tree,
            self.node_components,
            point_components,
            self.core_distances,
            self.noise_fraction,
        )


@numba.njit(parallel=True)
def boruvka_tree_query(
    tree, node_components, point_components, core_distances, noise_fraction
):
    """Finds the closest neighbor in another component for each data point."""
    candidate_distances = np.full((tree.data.shape[0], 1), np.inf, dtype=np.float32)
    candidate_indices = np.full((tree.data.shape[0], 1), -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(
        tree.data.shape[0], np.inf, dtype=np.float32
    )

    data = np.asarray(tree.data.astype(np.float32))
    for i in numba.prange(tree.data.shape[0]):
        distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i]
        )
        heap_p, heap_i = candidate_distances[i], candidate_indices[i]
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

    return deheap_sort(candidate_distances, candidate_indices)


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
    if _component_aware_stop_condition(
        node,
        heap_p,
        current_core_distance,
        current_component,
        node_components,
        dist_lower_bound,
        component_nearest_neighbor_dist,
    ):
        return

    if tree.node_data[node].is_leaf:
        _component_aware_update(
            tree,
            node,
            point,
            heap_p,
            heap_i,
            current_core_distance,
            core_distances,
            current_component,
            point_components,
            component_nearest_neighbor_dist,
            noise_fraction,
        )
    else:
        _component_aware_traversal(
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
            component_nearest_neighbor_dist,
            noise_fraction,
        )


@numba.njit()
def _component_aware_update(
    tree,
    node,
    point,
    heap_p,
    heap_i,
    current_core_distance,
    core_distances,
    current_component,
    point_components,
    component_nearest_neighbor_dist,
    noise_fraction,
):
    node_info = tree.node_data[node]
    for i in range(node_info.idx_start, node_info.idx_end):
        idx = tree.idx_array[i]
        if (
            point_components[idx] != current_component
            and core_distances[idx] < component_nearest_neighbor_dist[0]
        ):
            mutual_core = max(current_core_distance, core_distances[idx])
            d = max(
                rdist(point, tree.data[idx])
                + np.random.normal(0.0, noise_fraction * mutual_core, 1)[0],
                0.0,
            )
            if d < heap_p[0]:
                simple_heap_push(heap_p, heap_i, d, idx)
                if heap_p[0] < component_nearest_neighbor_dist[0]:
                    component_nearest_neighbor_dist[0] = heap_p[0]


@numba.njit()
def _component_aware_traversal(
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
    component_nearest_neighbor_dist,
    noise_fraction,
):
    # ------------------------------------------------------------
    # Case 3: Node is not a leaf. Recursively query subnodes starting
    #         with the closest
    left = 2 * node + 1
    right = left + 1
    dist_lower_bound_left = point_to_node_lower_bound_rdist(
        tree.node_bounds[0, left], tree.node_bounds[1, left], point
    )
    dist_lower_bound_right = point_to_node_lower_bound_rdist(
        tree.node_bounds[0, right], tree.node_bounds[1, right], point
    )

    if dist_lower_bound_left <= dist_lower_bound_right:
        side_1, side_2 = left, right
        bound_1, bound_2 = dist_lower_bound_left, dist_lower_bound_right
    else:
        side_1, side_2 = right, left
        bound_1, bound_2 = dist_lower_bound_right, dist_lower_bound_left

    _component_aware_query_recursion(
        tree,
        side_1,
        point,
        heap_p,
        heap_i,
        current_core_distance,
        core_distances,
        current_component,
        node_components,
        point_components,
        bound_1,
        component_nearest_neighbor_dist,
        noise_fraction,
    )
    _component_aware_query_recursion(
        tree,
        side_2,
        point,
        heap_p,
        heap_i,
        current_core_distance,
        core_distances,
        current_component,
        node_components,
        point_components,
        bound_2,
        component_nearest_neighbor_dist,
        noise_fraction,
    )
