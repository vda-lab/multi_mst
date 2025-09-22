import numba
import numpy as np
from sklearn.utils import check_random_state
from scipy.spatial.distance import squareform
from pynndescent.pynndescent_ import INT32_MIN, INT32_MAX
from pynndescent.utils import (
    tau_rand_int,
    make_heap,
    deheap_sort,
    simple_heap_push,
    checked_heap_push,
)

from .descent import (
    _sample_in_candidates,
    _sample_out_candidates,
    _apply_graph_updates,
    _group_indices_per_component,
)


class PrecomputedIndex(object):
    def __init__(
        self,
        distances,
        num_neighbors=5,
        min_samples=1,
        min_descent_neighbors=12,
        nn_kwargs=None,
    ):
        if nn_kwargs is None:
            nn_kwargs = {}

        self.distances = distances
        self.num_points = distances.shape[0]
        self.num_neighbors = num_neighbors
        self.min_samples = min_samples
        self.n_threads = numba.get_num_threads()
        self.descent_neighbors = max(num_neighbors + 1, min_descent_neighbors)
        self.n_iters = nn_kwargs.get(
            "n_iters", max(5, int(round(np.log2(self.num_points))))
        )
        self.delta = nn_kwargs.get("delta", 0.001)
        self.rng_state = nn_kwargs.get(
            "rng_state",
            check_random_state(None).randint(INT32_MIN, INT32_MAX, 3).astype(np.int64),
        )

    def neighbors(self):
        neighbors = np.argpartition(
            self.distances, np.arange(self.num_neighbors + 1), axis=1
        )[:, : self.num_neighbors + 1]
        distances = np.take_along_axis(self.distances, neighbors, axis=1)

        self.in_graph = neighbors[:, 1:]
        self._neighbors = neighbors[:, 1:]
        self.core_distances = distances.T[self.min_samples]
        self._distances = np.maximum(
            distances[:, 1:],
            np.maximum(
                self.core_distances[:, None],
                self.core_distances[self._neighbors],
            ),
        )

        return (
            distances[:, : self.num_neighbors + 1],
            neighbors[:, : self.num_neighbors + 1],
        )

    def query(self, point_components):
        heap_graph, remapped_components = initialize_out_graph(
            self.distances,
            self.core_distances,
            (self._neighbors, self._distances),
            point_components,
            self.rng_state,
        )
        precomputed_descent(
            self.distances,
            self.core_distances,
            self.in_graph,
            heap_graph,
            remapped_components,
            min(60, self.descent_neighbors),
            3 * self.n_iters,
            self.delta,
            self.rng_state,
            self.n_threads,
        )

        self._neighbors, self._distances = deheap_sort(*heap_graph[:2])
        return self._distances, self._neighbors

    def correction(self, distances):
        return distances


@numba.njit(
    parallel=True,
    locals={
        "cnt": numba.int32,
        "idx": numba.int32,
        "size": numba.int32,
        "k": numba.int32,
        "d": numba.float32,
    },
)
def initialize_out_graph(distances, core_distances, graph, point_components, rng_state):
    """Replaces neighbors in the same component with random points from other components."""
    # Create empty heap to size
    grouped_indices, remapped_components = _group_indices_per_component(
        point_components
    )
    descent_neighbors = graph[0].shape[1]
    new_graph = make_heap(distances.shape[0], descent_neighbors)

    # Fill the new graph
    for i in numba.prange(distances.shape[0]):
        # Copy points from old graph that are not in the same component
        cnt = 0
        for j, d in zip(graph[0][i], graph[1][i]):
            if j < 0 or remapped_components[i] == remapped_components[j]:
                continue
            simple_heap_push(
                new_graph[1][i],
                new_graph[0][i],
                d,
                j,
            )
            cnt += 1

        # Fill remaining slots with random points in other components
        tries = 0
        num_points_in_comp = len(grouped_indices[remapped_components[i]])
        while cnt < descent_neighbors and tries < 2 * descent_neighbors:
            tries += 1

            # Sample random number in range [0, num-points-not-in-same-comp)
            idx = np.abs(tau_rand_int(rng_state)) % (
                distances.shape[0] - num_points_in_comp
            )

            # Find the idx-th not-in-same-comp data point index
            for k, indices in enumerate(grouped_indices):
                if k == remapped_components[i]:
                    continue
                size = np.int32(len(indices))
                if idx >= size:
                    idx -= size
                else:
                    idx = indices[idx]
                    break

            # Add idx to i's neighbors
            d = max(distances[idx, i], core_distances[idx], core_distances[i])
            cnt += checked_heap_push(new_graph[1][i], new_graph[0][i], d, idx)

    # Set all flags to true
    new_graph[2][:] = np.uint8(1)
    return new_graph, remapped_components


@numba.njit(cache=True)
def precomputed_descent(
    distances,
    core_distances,
    in_graph,
    out_graph,
    point_components,
    max_candidates,
    n_iters,
    delta,
    rng_state,
    n_threads,
):
    """Runs NN Descent variant looking for nearest neighbors in other components.

    Updates are more like the initially described algorithm than the local join
    algorithm. We keep track of two graphs:
        - the in-graph contains normal nearest neighbors and remains fixed.
        - the out-graph is updated to contain the nearest neighbors in other components.

    The update step samples neighbors in the out-graph (both directions) compares their
    in-graph neighbors to find nearer neighbors in other components.
    """
    for _ in range(n_iters):
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
            distances,
            core_distances,
            point_components,
            out_graph[1][:, 0],
            in_neighbors,
            out_neighbors,
        )
        # Update the out-graph.
        c = _apply_graph_updates(out_graph, updates, n_threads)
        # Early termination
        if c <= delta * in_graph.shape[1] * distances.shape[0]:
            break


@numba.njit(parallel=True, cache=True)
def _generate_graph_updates(
    distances,
    core_distances,
    point_components,
    dist_thresholds,
    in_neighbors,
    out_neighbors,
):
    n_vertices = out_neighbors.shape[0]
    updates = [[(-1, np.inf) for _ in range(0)] for _ in range(n_vertices)]

    # Iterate over vertices
    for current_idx in numba.prange(n_vertices):
        # Iterate over their out-graph sample
        for neighbor_idx in out_neighbors[current_idx]:
            if neighbor_idx < 0:
                continue
            # Iterate over their in-graph neighbors
            for candidate_idx in in_neighbors[neighbor_idx]:
                if (
                    candidate_idx < 0
                    or point_components[candidate_idx] == point_components[current_idx]
                ):
                    # Need to check components differ because Descent may run on
                    # more neighbors than accepted by the MST! So the in-graph
                    # may contain neighbors not yet connected!
                    continue

                d = max(
                    distances[current_idx, candidate_idx],
                    core_distances[candidate_idx],
                    core_distances[current_idx],
                )
                if d <= max(
                    dist_thresholds[current_idx],
                    dist_thresholds[candidate_idx],
                ):
                    updates[current_idx].append((candidate_idx, d))

    return updates
