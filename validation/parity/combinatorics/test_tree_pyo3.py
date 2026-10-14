# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""PyO3 reachability and parity for the tree-core kernel (topic U1).

This is the Python-surface gate for `noether.combin.Tree`: it confirms
every accepted function is reachable from Python and that the PyO3 boundary
returns the same answers as an independent naive Python reference (root-path
LCA / ancestor walks / brute-force centroid / all-pairs BFS diameter) on a
spread of small trees.

It is deliberately self-contained: pure-Python references, **no** Sage / no
NetworkX / no oracle binary, so it always runs (a true reachability gate) and
does not duplicate the Rust-oracle Sage/NetworkX lane in
``test_tree_core_parity.py``.

Every accepted function appears in an asserted cell:
``from_edges`` (construction + non-tree rejection), ``lca``, ``depth``,
``kth_ancestor``, ``is_ancestor``, ``subtree_size``, ``subtree``,
``diameter``, ``centroid``, ``centroid_decomposition``,
``heavy_light_decomposition`` (+ ``hld_path_intervals``), and ``tree_dp``.
"""

from __future__ import annotations

from collections import deque

import pytest

ne = pytest.importorskip("noether.combin")


# ---------- tree corpus ----------


def _path(n):
    return [(i, i + 1) for i in range(n - 1)]


def _star(n):
    return [(0, i) for i in range(1, n)]


def _balanced_binary(n):
    return [((i - 1) // 2, i) for i in range(1, n)]


def _caterpillar(spine):
    n = 2 * spine
    edges = [(i, i + 1) for i in range(spine - 1)]
    edges += [(i, spine + i) for i in range(spine)]
    return n, edges


def _random_tree(n, seed):
    # Random recursive tree — attach i to a random earlier vertex (matches the
    # splitmix64 generator the Rust unit tests use, but any tree is fine here).
    state = seed & 0xFFFFFFFFFFFFFFFF
    edges = []

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    for i in range(1, n):
        edges.append((nxt() % i, i))
    return edges


def _corpus():
    cases = [("single", 1, [])]
    for n in (2, 8, 33):
        cases.append((f"path-{n}", n, _path(n)))
        cases.append((f"star-{n}", n, _star(n)))
        cases.append((f"balbin-{n}", n, _balanced_binary(n)))
        for s in range(2):
            cases.append((f"rand-{n}-{s}", n, _random_tree(n, 0xABC0 ^ (n << 8) ^ s)))
    for spine in (4, 12):
        n, edges = _caterpillar(spine)
        cases.append((f"caterpillar-{spine}", n, edges))
    return cases


CORPUS = _corpus()


# ---------- naive Python references (rooted at 0) ----------


def _adj(n, edges):
    a = [[] for _ in range(n)]
    for u, v in edges:
        a[u].append(v)
        a[v].append(u)
    return a


def _root(n, a, root=0):
    """BFS rooting → (parent, depth). parent[root] = root."""
    parent = [root] * n
    depth = [-1] * n
    depth[root] = 0
    q = deque([root])
    while q:
        u = q.popleft()
        for w in a[u]:
            if depth[w] == -1:
                depth[w] = depth[u] + 1
                parent[w] = u
                q.append(w)
    return parent, depth


def _ancestors_path(v, parent, root=0):
    out = [v]
    while v != root:
        v = parent[v]
        out.append(v)
    return out  # [v, parent, ..., root]


def _naive_lca(u, v, parent, root=0):
    av = set(_ancestors_path(v, parent, root))
    for x in _ancestors_path(u, parent, root):
        if x in av:
            return x
    return root


def _naive_descendants(n, parent, v, root=0):
    out = []
    for x in range(n):
        cur = x
        while True:
            if cur == v:
                out.append(x)
                break
            if cur == root:
                break
            cur = parent[cur]
    return sorted(out)


def _naive_diameter(n, a):
    best = 0
    for s in range(n):
        depth = [-1] * n
        depth[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in a[u]:
                if depth[w] == -1:
                    depth[w] = depth[u] + 1
                    q.append(w)
        best = max(best, max(depth))
    return best


def _naive_centroids(n, a):
    out = []
    for c in range(n):
        # largest component left by removing c (BFS the rest).
        seen = [False] * n
        seen[c] = True
        maxpart = 0
        for s in range(n):
            if seen[s]:
                continue
            sz = 0
            stack = [s]
            seen[s] = True
            while stack:
                u = stack.pop()
                sz += 1
                for w in a[u]:
                    if not seen[w]:
                        seen[w] = True
                        stack.append(w)
            maxpart = max(maxpart, sz)
        if maxpart <= n // 2:
            out.append(c)
    return sorted(out)


# ---------- the gate ----------


@pytest.mark.parametrize("name,n,edges", CORPUS, ids=[c[0] for c in CORPUS])
def test_tree_pyo3_matches_naive(name, n, edges):
    a = _adj(n, edges)
    parent, depth = _root(n, a, 0)

    t = ne.Tree.from_edges(n, edges)  # from_edges / tree_from_edges
    assert t.order == n
    assert t.root == 0
    assert t.type_id == "noether/combin/Tree@1"

    verts = list(range(n))

    # depth
    for v in verts:
        assert t.depth(v) == depth[v], f"{name}: depth[{v}]"

    # lca + distance (all pairs for small n, a sample otherwise)
    pairs = (
        [(u, v) for u in verts for v in verts] if n <= 16 else [(u, (u * 7 + 3) % n) for u in verts]
    )
    for u, v in pairs:
        w = _naive_lca(u, v, parent, 0)
        assert t.lca(u, v) == w, f"{name}: lca({u},{v})"
        assert t.distance(u, v) == depth[u] + depth[v] - 2 * depth[w], f"{name}: distance({u},{v})"

    # subtree_size / subtree (preorder set) / is_ancestor
    for v in verts:
        desc = _naive_descendants(n, parent, v, 0)
        assert t.subtree_size(v) == len(desc), f"{name}: subtree_size[{v}]"
        assert sorted(t.subtree(v)) == desc, f"{name}: subtree[{v}]"
    for d in verts:
        anc = set(_ancestors_path(d, parent, 0))
        for x in verts:
            assert t.is_ancestor(x, d) == (x in anc), f"{name}: is_ancestor({x},{d})"

    # kth_ancestor
    for v in verts:
        path = _ancestors_path(v, parent, 0)  # path[k] = k-th ancestor
        for k in range(depth[v] + 3):
            exp = path[k] if k < len(path) else None
            assert t.kth_ancestor(v, k) == exp, f"{name}: kth_ancestor({v},{k})"

    # diameter (length + a valid realizing path)
    diam, dpath = t.diameter()
    assert diam == _naive_diameter(n, a), f"{name}: diameter length"
    assert len(dpath) == diam + 1, f"{name}: diameter path length"
    eset = {tuple(sorted(e)) for e in edges}
    for x, y in zip(dpath, dpath[1:]):
        assert tuple(sorted((x, y))) in eset, f"{name}: diameter path edge"

    # centroid (size-balance) vs brute force
    assert t.centroid() == _naive_centroids(n, a), f"{name}: centroid"

    # centroid_decomposition — a real tree (one root, O(log n) depth)
    cd = t.centroid_decomposition()
    parents = cd["parent"]
    assert parents.count(None) == 1, f"{name}: exactly one cd-root"
    assert cd["root"] in verts and parents[cd["root"]] is None
    assert cd["depth"] <= 2 * max(1, (n).bit_length()), f"{name}: cd depth O(log n)"

    # heavy_light_decomposition — positions are a permutation of 0..n
    hld = t.heavy_light_decomposition()
    assert sorted(hld["position"]) == verts, f"{name}: HLD positions a permutation"
    assert len(hld["head"]) == n and len(hld["chain"]) == n
    # hld_path_intervals: the disjoint union of intervals covers exactly the
    # u‥v tree path (checked against the naive root-path-derived path set).
    if n >= 2:
        u, v = 0, n - 1
        # naive vertex set on the u‥v path via LCA
        w = _naive_lca(u, v, parent, 0)
        path_set = set(_ancestors_path(u, parent, 0)[: depth[u] - depth[w] + 1])
        path_set |= set(_ancestors_path(v, parent, 0)[: depth[v] - depth[w] + 1])
        pos_to_vertex = {hld["position"][x]: x for x in verts}
        covered = set()
        for lo, hi in t.hld_path_intervals(u, v):
            for p in range(lo, hi + 1):
                covered.add(pos_to_vertex[p])
        assert covered == path_set, f"{name}: HLD path intervals cover the path"

    # tree_dp — the subtree-size DP equals subtree_size on every vertex
    dp = t.tree_dp()
    assert dp == [t.subtree_size(v) for v in verts], f"{name}: tree_dp = subtree sizes"


def test_tree_pyo3_rejects_non_trees():
    """A non-tree raises a typed construction error."""
    # A 3-cycle: connected but n edges, not n-1 → not a tree.
    with pytest.raises(ValueError):
        ne.Tree.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    # Disconnected (a forest): wrong edge count for a single tree.
    with pytest.raises(ValueError):
        ne.Tree.from_edges(4, [(0, 1), (2, 3)])
    # Root out of range → IndexError.
    with pytest.raises(IndexError):
        ne.Tree.from_edges(3, [(0, 1), (1, 2)], 9)
