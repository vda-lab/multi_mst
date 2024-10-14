# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Adapted for MST descent by Jelmer Bot <jelmer.bot@uhasselt.be>
# License: BSD 2 clause
import numba
import numpy as np
from pynndescent.utils import (
    tau_rand,
    tau_rand_int,
    make_heap,
    deheap_sort,
    simple_heap_push,
    checked_heap_push,
    checked_flagged_heap_push,
)


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
    data, dist, graph, remaped_components, grouped_unique, rng_state
):
    """Replaces neighbours in the same component with random points from other components."""
    # Create empty heap to size
    descent_neighbours = graph[0].shape[1]
    new_graph = make_heap(data.shape[0], descent_neighbours)

    # Fill the new graph
    for i in numba.prange(data.shape[0]):
        # Copy points from old graph that are not in the same component
        cnt = 0
        for j, d in zip(graph[0][i], graph[1][i]):
            if j < 0 or remaped_components[i] == remaped_components[j]:
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
        num_points_in_comp = grouped_unique[remaped_components[i]].shape[0]
        while cnt < descent_neighbours and tries < 2 * descent_neighbours:
            tries += 1
            
            # Sample random number in range [0, num-points-not-in-same-comp)
            idx = np.abs(tau_rand_int(rng_state)) % (data.shape[0] - num_points_in_comp)

            # Find the idx-th not-in-same-comp data point index
            for k, indices in enumerate(grouped_unique):
                if k == remaped_components[i]:
                    continue
                size = np.int32(indices.shape[0])
                if idx >= size:
                    idx -= size
                else:
                    idx = indices[idx]
                    break

            # Add idx to i's neighbors
            d = dist(data[idx], data[i])
            cnt += checked_heap_push(new_graph[1][i], new_graph[0][i], d, idx)

    # Set all flags to true
    new_graph[2][:] = np.uint8(1)
    return new_graph


@numba.njit()
def nn_descent(
    data,
    dist,
    in_graph,
    out_graph,
    remapped_components,
    max_candidates,
    n_iters,
    delta,
    rng_state,
):
    """Runs NN Descent variant looking for nearest neighbors in other components.

    Updates are more like the initially described algorithm than the local join
    algorithm. We keep track of two graphs:
        - the in-graph contains normal nearest neighbors and remains fixed.
        - the out-graph is updated to contain the nearest neighbors in other components.

    The update step samples neighbors in the out-graph (both directions) compares their
    in-graph neighbors to find nearer neighbors in other components.
    """
    n_vertices = data.shape[0]
    n_threads = numba.get_num_threads()
    delta_flag = False

    for _ in range(n_iters):
        # Sample new (undirected) neighbours in the out-graph.
        out_neighbors = _sample_out_candidates(
            out_graph, max_candidates, rng_state, n_threads
        )
        # Direct neighbours + sampled reverse neighbours in the in-graph.
        in_neighbors = _sample_in_candidates(
            in_graph, max_candidates, rng_state, n_threads
        )
        # Find updates using the two sets of neighbors.
        updates = _generate_graph_updates(
            data,
            dist,
            remapped_components,
            out_graph[1][:, 0],
            in_neighbors,
            out_neighbors,
        )
        # Update the out-graph.
        c = _apply_graph_updates(out_graph, updates, n_threads)
        # Early termination
        if c <= delta * in_graph[0].shape[1] * data.shape[0]:
            delta_flag = True
            break

    deheap_sort(out_graph[0], out_graph[1])
    

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
                isn = current_flags[i, j] # is_new

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
    current_indices = in_graph[0]
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


@numba.njit(parallel=True)
def _generate_graph_updates(
    data,
    dist,
    remapped_components,
    dist_thresholds,
    in_neighbors,
    out_neighbors,
):
    n_vertices = out_neighbors.shape[0]
    max_candidates = out_neighbors.shape[1]
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
                    or remapped_components[candidate_idx]
                    == remapped_components[current_idx]
                ):
                    # Need to check components differ because Descent may run on
                    # more neighbors than accepted by the MST! So the in-graph
                    # may contain neighbors not yet connected!
                    continue

                d = dist(data[current_idx], data[candidate_idx])
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
