import numba
import numpy as np
from pynndescent import NNDescent
from pynndescent.utils import (
    tau_rand,
    tau_rand_int,
    make_heap,
    deheap_sort,
    simple_heap_push,
    checked_heap_push,
    checked_flagged_heap_push,
)


class DescentIndex(object):
    def __init__(
        self,
        data,
        metric="euclidean",
        metric_kwds=None,
        num_neighbors=5,
        min_samples=1,
        min_descent_neighbors=12,
        nn_kwargs=None,
    ):
        if metric_kwds is None:
            metric_kwds = {}
        if nn_kwargs is None:
            nn_kwargs = {}

        self.num_points = data.shape[0]
        self.num_neighbors = num_neighbors
        self.min_samples = min_samples
        self.n_threads = numba.get_num_threads()
        self.index = NNDescent(
            data,
            metric=metric,
            metric_kwds=metric_kwds,
            n_neighbors=max(num_neighbors + 1, min_descent_neighbors),
            **nn_kwargs,
        )

    def neighbors(self):
        neighbors, distances = self.index._neighbor_graph

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
            self.index._raw_data,
            self.index._distance_func,
            self.core_distances,
            (self._neighbors, self._distances),
            point_components,
            self.index.rng_state,
        )
        nn_descent(
            self.index._raw_data,
            self.index._distance_func,
            self.core_distances,
            self.in_graph,
            heap_graph,
            remapped_components,
            min(60, self.index.n_neighbors),
            3 * self.index.n_iters,
            self.index.delta,
            self.index.rng_state,
            self.n_threads,
        )

        self._neighbors, self._distances = deheap_sort(*heap_graph[:2])
        return self._distances, self._neighbors

    def correction(self, distances):
        if self.index._distance_correction is not None:
            return self.index._distance_correction(distances)
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
def initialize_out_graph(
    data, dist, core_distances, graph, point_components, rng_state
):
    """Replaces neighbors in the same component with random points from other components."""
    # Create empty heap to size
    grouped_indices, remapped_components = _group_indices_per_component(
        point_components
    )
    descent_neighbors = graph[0].shape[1]
    new_graph = make_heap(data.shape[0], descent_neighbors)

    # Fill the new graph
    for i in numba.prange(data.shape[0]):
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
            idx = np.abs(tau_rand_int(rng_state)) % (data.shape[0] - num_points_in_comp)

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
            d = max(dist(data[idx], data[i]), core_distances[idx], core_distances[i])
            cnt += checked_heap_push(new_graph[1][i], new_graph[0][i], d, idx)

    # Set all flags to true
    new_graph[2][:] = np.uint8(1)
    return new_graph, remapped_components


@numba.njit(cache=True)
def nn_descent(
    data,
    dist,
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
        # Early termination
        if c <= delta * in_graph.shape[1] * data.shape[0]:
            break


@numba.njit(parallel=True, locals={"idx": numba.types.int64}, cache=True)
def _sample_out_candidates(out_graph, max_candidates, rng_state, n_threads):
    current_indices = out_graph[0]
    current_flags = out_graph[2]
    n_vertices = current_indices.shape[0]
    n_neighbors = current_indices.shape[1]

    shape = (n_vertices, max_candidates)
    candidate_indices = np.full(shape, -1, dtype=np.int32)
    candidate_priority = np.full(shape, np.inf, dtype=np.float32)

    # Sample from new neighbors in the out-graph.
    for n in numba.prange(n_threads):
        local_rng_state = rng_state + n
        for i in range(n_vertices):

            for j in range(n_neighbors):
                idx = current_indices[i, j]
                isn = current_flags[i, j]  # is_new

                if idx < 0 or not isn:
                    continue

                d = tau_rand(local_rng_state)

                if i % n_threads == n:
                    checked_heap_push(
                        candidate_priority[i], candidate_indices[i], d, idx
                    )
                if idx % n_threads == n:
                    checked_heap_push(
                        candidate_priority[idx],
                        candidate_indices[idx],
                        d,
                        i,
                    )

    # Update flags
    indices, _, flags = out_graph
    for i in numba.prange(n_vertices):
        for j in range(n_neighbors):
            idx = indices[i, j]

            for k in range(max_candidates):
                if candidate_indices[i, k] == idx:
                    flags[i, j] = 0
                    break

    return candidate_indices


@numba.njit(parallel=True, locals={"idx": numba.types.int64}, cache=True)
def _sample_in_candidates(in_graph, max_candidates, rng_state, n_threads):
    current_indices = in_graph
    n_vertices = current_indices.shape[0]
    n_neighbors = current_indices.shape[1]

    shape = (n_vertices, max_candidates)
    candidate_indices = np.full(shape, -1, dtype=np.int32)
    candidate_priority = np.full(shape, np.inf, dtype=np.float32)

    # Sample reverse neighbors in the in-graph.
    for n in numba.prange(n_threads):
        local_rng_state = rng_state + n
        for i in range(n_vertices):
            for j in range(n_neighbors):
                idx = current_indices[i, j]
                if idx < 0:
                    continue

                if i % n_threads == n:
                    checked_heap_push(
                        candidate_priority[i], candidate_indices[i], 0, idx
                    )
                if idx % n_threads == n:
                    checked_heap_push(
                        candidate_priority[idx],
                        candidate_indices[idx],
                        tau_rand(local_rng_state),
                        i,
                    )
    return candidate_indices


@numba.njit(parallel=True, cache=True)
def _generate_graph_updates(
    data,
    dist,
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
                    dist(data[current_idx], data[candidate_idx]),
                    core_distances[candidate_idx],
                    core_distances[current_idx],
                )
                if d <= max(
                    dist_thresholds[current_idx],
                    dist_thresholds[candidate_idx],
                ):
                    updates[current_idx].append((candidate_idx, d))

    return updates


@numba.njit(
    parallel=True,
    locals={
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "added": numba.uint8,
        "n": numba.uint32,
        "current_idx": numba.uint32,
        "j": numba.uint32,
    },
    cache=True,
)
def _apply_graph_updates(out_graph, updates, n_threads):
    n_changes = 0

    priorities = out_graph[1]
    indices = out_graph[0]
    flags = out_graph[2]

    for n in numba.prange(n_threads):
        for current_idx in range(len(updates)):
            for j in range(len(updates[current_idx])):
                neighbor_idx, d = updates[current_idx][j]

                if current_idx % n_threads == n:
                    added = checked_flagged_heap_push(
                        priorities[current_idx],
                        indices[current_idx],
                        flags[current_idx],
                        d,
                        neighbor_idx,
                        1,
                    )
                    n_changes += added

                if neighbor_idx % n_threads == n:
                    added = checked_flagged_heap_push(
                        priorities[neighbor_idx],
                        indices[neighbor_idx],
                        flags[neighbor_idx],
                        d,
                        current_idx,
                        1,
                    )
                    n_changes += added

    return n_changes


@numba.njit(parallel=True, cache=True)
def _group_indices_per_component(point_components):
    # Numba does not support lists in dictionaries, so use this instead
    sort_idx = np.argsort(point_components)
    sorted = point_components[sort_idx]
    unq_first = np.empty(sorted.shape, dtype=np.bool_)
    unq_first[0] = True
    unq_first[1:] = sorted[1:] != sorted[:-1]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    grouped_indices = []
    start = 0
    for count in unq_count:
        grouped_indices.append(sort_idx[start : start + count])
        start += count
    grouped_indices.append(sort_idx[start:])

    remapped = np.empty_like(point_components)
    for i in numba.prange(len(grouped_indices)):
        remapped[grouped_indices[i]] = i

    return grouped_indices, remapped
