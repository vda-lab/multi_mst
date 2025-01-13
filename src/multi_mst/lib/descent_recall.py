import numba
import numpy as np

from .descent import (
    DescentIndex,
    deheap_sort,
    initialize_out_graph,
    _apply_graph_updates,
    _generate_graph_updates,
    _sample_in_candidates,
    _sample_out_candidates,
)


class DescentRecallIndex(DescentIndex):
    def query(self, point_components, true_candidates, true_selected):
        heap_graph, remapped_components = initialize_out_graph(
            self.index._raw_data,
            self.index._distance_func,
            self.core_distances,
            (self._neighbors, self._distances),
            point_components,
            self.index.rng_state,
        )

        recall_trace, dist_frac_trace, change_trace = nn_descent(
            self.index._raw_data,
            self.index._distance_func,
            self.core_distances,
            self.in_graph,
            heap_graph,
            true_candidates,
            true_selected,
            remapped_components,
            min(60, self.index.n_neighbors),
            3 * self.index.n_iters,
            self.index.delta,
            self.index.rng_state,
            self.n_threads,
        )

        self._neighbors, self._distances = deheap_sort(*heap_graph[:2])
        return (
            self._distances,
            self._neighbors,
            (recall_trace, dist_frac_trace, change_trace),
        )


@numba.njit()
def nn_descent(
    data,
    dist,
    core_distances,
    in_graph,
    out_graph,
    true_candidates,
    true_selected,
    point_components,
    max_candidates,
    n_iters,
    delta,
    rng_state,
    n_threads,
):
    """Runs NN Descent variant looking for nearest neighbors in other components
    while tracking recall given true nearest neighbors.

    Updates are more like the initially described algorithm than the local join
    algorithm. We keep track of two graphs:
        - the in-graph contains normal nearest neighbors and remains fixed.
        - the out-graph is updated to contain the nearest neighbors in other components.

    The update step samples neighbors in the out-graph (both directions) compares their
    in-graph neighbors to find nearer neighbors in other components.
    """
    # Allocate traces
    change_trace = np.full(n_iters, np.nan, dtype=np.float32)
    recall_trace = np.full(n_iters, np.nan, dtype=np.float32)
    dist_frac_trace = np.full(n_iters, np.nan, dtype=np.float32)

    # Extract points with ground-truth edges added this iteration
    true_selected_points = np.sort(true_selected[1].flatten())
    true_selected_points = true_selected_points[np.argmax(true_selected_points >= 0) :]
    true_total_distance = _compute_dist_total(true_selected_points, true_candidates[0])

    for i in range(n_iters):
        # Sample new (undirected) neighbors in the out-graph.
        out_neighbors = _sample_out_candidates(
            out_graph, max_candidates, rng_state, n_threads
        )
        # Direct neighbors + sampled reverse neighbors in the in-graph.
        in_neighbors = _sample_in_candidates(
            in_graph, max_candidates, rng_state, n_threads
        )
        # Find updates using the two sets of neighbors.
        updates = _generate_graph_updates(
            data,
            dist,
            core_distances,
            point_components,
            out_graph[1][:, 0],
            in_neighbors,
            out_neighbors,
        )
        # Update the out-graph.
        c = _apply_graph_updates(out_graph, updates, n_threads)

        # Fill traces
        out_graph_copy = deheap_sort(out_graph[0].copy(), out_graph[1].copy())
        change_trace[i] = c
        recall_trace[i] = _compute_recall(true_candidates, out_graph_copy)
        dist_frac_trace[i] = (
            _compute_dist_total(true_selected_points, out_graph_copy[1])
            / true_total_distance
        )

        # Early termination
        if c <= delta * in_graph.shape[1] * data.shape[0]:
            break

    return recall_trace, dist_frac_trace, change_trace


@numba.njit(cache=True)
def _compute_recall(truth, current):
    num_points = truth[0].shape[0]
    num_neighbors = truth[0].shape[1]

    # Compute intersection (sparse multiplication)
    true_positives = 0
    for pt in range(num_points):
        i1 = 0
        i2 = 0
        while max(i1, i2) < num_neighbors:
            t_d = truth[0][pt, i1]  # true distance
            t_t = truth[1][pt, i1]  # true target
            p_d = current[1][pt, i2]  # predicted distances
            p_t = current[0][pt, i2]  # predicted target
            if min(t_t, p_t) < 0:
                break
            if t_t == p_t:
                true_positives += 1
                i1 += 1
                i2 += 1
            elif t_d < p_d:
                i1 += 1
            else:
                i2 += 1
    return true_positives / np.sum(truth[1] >= 0)


@numba.njit(cache=True)
def _compute_dist_total(true_selected_points, dists):
    distance_total = 0

    i1 = 0
    i2 = 0
    pt_cnt = 0
    while i1 < true_selected_points.shape[0] and i2 < dists.shape[0]:
        if i2 == true_selected_points[i1]:
            distance_total += dists[i2, pt_cnt]
            i1 += 1
            pt_cnt += 1
        else:
            i2 += 1
            pt_cnt = 0
    return distance_total
