import numba
import numpy as np
from .disjoint_set import ds_find


@numba.njit(parallel=True)
def update_component_vectors(tree, disjoint_set, node_components, point_components):
    """Updates node and point level component IDs from the disjoint set."""
    for i in numba.prange(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))

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
