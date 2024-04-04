import numba
import numpy as np

from collections import namedtuple

RankDisjointSet = namedtuple("DisjointSet", ["parent", "rank"])
SizeDisjointSet = namedtuple("DisjointSet", ["parent", "size"])


@numba.njit()
def ds_rank_create(n_elements):
    """Initialises UnionFind structure with ranks."""
    return RankDisjointSet(np.arange(n_elements, dtype=np.int32), np.zeros(n_elements, dtype=np.int32))

@numba.njit()
def ds_rank_copy(disjoint_set):
    """Creates a copy of the disjoint set."""
    return RankDisjointSet(disjoint_set.parent.copy(), disjoint_set.rank.copy())


@numba.njit()
def ds_size_create(n_elements):
    """Initialises UnionFind structure with sizes."""
    return SizeDisjointSet(np.arange(n_elements, dtype=np.int32), np.ones(n_elements, dtype=np.int32))


@numba.njit()
def ds_find(disjoint_set, x):
    """
    Finds the most recent parent of x in the disjoint set.Updates parents to
    grandparents inplace if they are present to speed up future queries
    """
    while disjoint_set.parent[x] != x:
        x, disjoint_set.parent[x] = disjoint_set.parent[x], disjoint_set.parent[disjoint_set.parent[x]]

    return x


@numba.njit()
def ds_union_by_rank(disjoint_set, x, y):
    """
    Looks for parents of two points and merges the lower rank one into the
    higher rank one.
    """
    x = ds_find(disjoint_set, x)
    y = ds_find(disjoint_set, y)

    if x == y:
        return

    if disjoint_set.rank[x] < disjoint_set.rank[y]:
        x, y = y, x

    disjoint_set.parent[y] = x
    if disjoint_set.rank[x] == disjoint_set.rank[y]:
        disjoint_set.rank[x] += 1


@numba.njit()
def ds_union_by_size(disjoint_set, x, y):
    """
    Looks for parents of two points and merges the smaller one into the larger
    one.
    """
    x = ds_find(disjoint_set, x)
    y = ds_find(disjoint_set, y)

    if x == y:
        return

    if disjoint_set.size[x] < disjoint_set.size[y]:
        x, y = y, x

    disjoint_set.parent[y] = x
    disjoint_set.size[x] += disjoint_set.size[y]