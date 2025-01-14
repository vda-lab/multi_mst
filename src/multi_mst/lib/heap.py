import numba

@numba.njit(
    [
        "void(f4[::1],i4[::1],i4[::1],f4,i4,i4)",
        "void(f8[::1],i4[::1],i4[::1],f8,i4,i4)",
    ],
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
)
def simple_edge_heap_push(priorities, sources, targets, p, s, t):
    """Inserts value (source, target) in to priority heap (distance)."""
    size = priorities.shape[0]

    # insert val at position zero
    priorities[0] = p
    sources[0] = s
    targets[0] = t

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        sources[i] = sources[i_swap]
        targets[i] = targets[i_swap]

        i = i_swap

    priorities[i] = p
    sources[i] = s
    targets[i] = t


@numba.njit()
def siftdown(heap1, heap2, heap3, elt):
    """Moves the element at index elt to its correct position in a heap."""
    while elt * 2 + 1 < heap1.shape[0]:
        left_child = elt * 2 + 1
        right_child = left_child + 1
        swap = elt

        if heap1[swap] < heap1[left_child]:
            swap = left_child

        if right_child < heap1.shape[0] and heap1[swap] < heap1[right_child]:
            swap = right_child

        if swap == elt:
            break
        else:
            heap1[elt], heap1[swap] = heap1[swap], heap1[elt]
            heap2[elt], heap2[swap] = heap2[swap], heap2[elt]
            heap3[elt], heap3[swap] = heap3[swap], heap3[elt]
            elt = swap


@numba.njit(parallel=True)
def deheap_sort_edges(distances, sources, targets):
    """Sorts the heaps and returns the sorted distances and indices."""
    for i in numba.prange(distances.shape[0]):
        # starting from the end of the array and moving back
        for j in range(sources.shape[1] - 1, 0, -1):
            sources[i, 0], sources[i, j] = sources[i, j], sources[i, 0]
            targets[i, 0], targets[i, j] = targets[i, j], targets[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]

            siftdown(distances[i, :j], sources[i, :j], targets[i, :j], 0)

    return distances, sources, targets

