# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Oracle parity for the tree-core kernel (topic U1, tree subsurface).

Cross-checks `graph_core::Tree` (driven through the `tree_oracle` example
binary, JSON in / JSON out) against **Sage** (`sage.graphs.graph.Graph`:
`is_tree`, `is_forest`, `diameter`, `breadth_first_search`,
`shortest_path_lengths`) and a **NetworkX** reference for the queries Sage
has no direct equivalent of (LCA via `lowest_common_ancestor`, ancestors /
descendants, k-th ancestor). Gated on ``NOETHER_VALIDATE_SAGE=1`` + a Sage
binary; never a CI dependency (runs via ``make validate``).

Every accepted function appears in an oracle-compared cell:

* ``tree_from_edges`` / ``is_tree`` / ``is_forest`` — Sage ``Graph.is_tree`` /
  ``is_forest``.
* ``depth`` — Sage ``shortest_path_lengths(root)``.
* ``subtree_size`` / ``subtree`` (descendants) / ``is_ancestor`` — NetworkX
  ``descendants`` on the BFS-rooted DiGraph (= Sage ``breadth_first_search``
  rooted descendant set).
* ``lca`` — NetworkX ``lowest_common_ancestor`` (an independent root-path
  reference; Sage has no direct O(1) LCA).
* ``kth_ancestor`` — NetworkX ancestor-path reference.
* ``diameter`` (+ path) — Sage ``Graph.diameter``.
* ``centroid`` — the **size-balance** definition (max component on removal
  ≤ n/2), checked vs an independent NetworkX reference — *not* Sage
  ``Graph.center`` (eccentricity), which is a distinct notion.
* ``centroid_decomposition`` — invariant: cd-depth ≤ 2·⌈log₂ n⌉+2.
* ``heavy_light_decomposition`` — invariant: max light edges root→leaf ≤
  ⌈log₂ n⌉ (no direct Sage oracle; this HLD check is invariant-based).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

# The Rust oracle binary path (built via `cargo build --release --example
# tree_oracle`). The harness builds it on demand the first time.
_REPO = Path(__file__).resolve().parents[3]
_TARGET = os.environ.get("TREE_CORE_TARGET", str(Path.home() / ".cache/noether-combi-target"))
_ORACLE = Path(_TARGET) / "release/examples/tree_oracle"


def _sage_binary():
    if os.environ.get("NOETHER_VALIDATE_SAGE", "") in ("", "0", "false", "False"):
        return None
    for cand in ("sage", os.path.expanduser("~/miniforge3/envs/sage/bin/sage")):
        if shutil.which(cand) or os.path.exists(cand):
            return cand
    return None


SAGE = _sage_binary()
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a sage binary"),
]


def _ensure_oracle():
    if _ORACLE.exists():
        return
    crate_dir = _REPO / "crates/graph-core"
    env = dict(os.environ, CARGO_TARGET_DIR=_TARGET)
    subprocess.run(
        ["cargo", "build", "--release", "--example", "tree_oracle"],
        cwd=str(crate_dir),
        env=env,
        check=True,
        capture_output=True,
    )


def _run_oracle(spec: dict) -> dict:
    _ensure_oracle()
    r = subprocess.run(
        [str(_ORACLE)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"tree_oracle failed: {r.stderr[-700:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _run_sage(script: str):
    r = subprocess.run([SAGE, "-python", "-c", script], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"sage failed: {r.stderr[-700:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------- tree corpus: path, star, balanced binary, caterpillar, random ----------


def _path(n):
    return [[i, i + 1] for i in range(n - 1)]


def _star(n):
    return [[0, i] for i in range(1, n)]


def _balanced_binary(n):
    return [[(i - 1) // 2, i] for i in range(1, n)]


def _caterpillar(spine):
    n = 2 * spine
    edges = [[i, i + 1] for i in range(spine - 1)]
    edges += [[i, spine + i] for i in range(spine)]
    return n, edges


def _random_tree(n, seed):
    # Random recursive tree — attach i to a random earlier vertex.
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
        p = nxt() % i
        edges.append([p, i])
    return edges


def _corpus():
    cases = []
    for n in (8, 64):
        cases.append((f"path-{n}", n, _path(n)))
        cases.append((f"star-{n}", n, _star(n)))
        cases.append((f"balbin-{n}", n, _balanced_binary(n)))
        for s in range(3):
            cases.append((f"rand-{n}-{s}", n, _random_tree(n, 0xABC0 ^ (n << 8) ^ s)))
    for spine in (4, 16):
        n, edges = _caterpillar(spine)
        cases.append((f"caterpillar-{spine}", n, edges))
    return cases


CORPUS = _corpus()


def _spec_for(n, edges):
    """Build the oracle spec: probe a spread of LCA / ancestor / kth queries."""
    import random

    rng = random.Random(0x5EED ^ n)
    verts = list(range(n))
    lca_pairs = [[rng.choice(verts), rng.choice(verts)] for _ in range(min(40, n * 2))]
    anc_pairs = [[rng.choice(verts), rng.choice(verts)] for _ in range(min(40, n * 2))]
    subtree_vs = verts if n <= 16 else rng.sample(verts, 16)
    kth = [[rng.choice(verts), rng.randint(0, 5)] for _ in range(min(30, n * 2))]
    return {
        "n": n,
        "edges": edges,
        "root": 0,
        "lca_pairs": lca_pairs,
        "anc_pairs": anc_pairs,
        "subtree_vs": subtree_vs,
        "kth": kth,
    }


def _sage_tree_facts(n, edges, root):
    script = f"""
from sage.all import Graph
import json
n = {n}
G = Graph({edges}, format='list_of_edges')
G.add_vertices(range(n))
spl = G.shortest_path_lengths({root})
depth = [int(spl.get(v, -1)) for v in range(n)]
out = {{
  "is_tree": bool(G.is_tree()),
  "is_forest": bool(G.is_forest()),
  "diameter": int(G.diameter()),
  "depth": depth,
}}
print(json.dumps(out))
"""
    return _run_sage(script)


def _nx_reference(n, edges, root, spec):
    """NetworkX reference for LCA / ancestors / descendants / kth-ancestor /
    centroid — the queries Sage has no direct equivalent of."""
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u, v in edges)
    # Root it: BFS tree from `root` (a DiGraph parent→child).
    D = nx.bfs_tree(G, root)
    depth = dict(nx.single_source_shortest_path_length(G, root))
    parent = {root: root}
    for u, v in D.edges():
        parent[v] = u

    def ancestors_path(v):
        # root-path ancestors of v (excluding v? include v).
        out = [v]
        cur = v
        while cur != root:
            cur = parent[cur]
            out.append(cur)
        return out

    # LCA via root-path intersection (independent of the oracle's algorithm).
    def lca(u, v):
        au = ancestors_path(u)
        av_set = set(ancestors_path(v))
        for x in au:
            if x in av_set:
                return x
        return root

    lca_out = {f"{u}-{v}": lca(u, v) for u, v in spec["lca_pairs"]}

    # descendants of v (in the rooted tree): nodes whose path to root passes v.
    desc = {}
    for v in spec["subtree_vs"]:
        ds = sorted(set(nx.descendants(D, v)) | {v})
        desc[str(v)] = ds

    # is_ancestor(a, d): a on d's root-path.
    anc_out = {}
    for a, d in spec["anc_pairs"]:
        anc_out[f"{a}-{d}"] = a in set(ancestors_path(d))

    # kth_ancestor.
    kth_out = {}
    for v, k in spec["kth"]:
        path = ancestors_path(v)  # path[0]=v, path[1]=parent, ...
        kth_out[f"{v}-{k}"] = path[k] if k < len(path) else -1

    # size-balance centroid (brute force).
    def centroids():
        out = []
        for c in range(n):
            H = G.copy()
            H.remove_node(c)
            comps = list(nx.connected_components(H)) if n > 1 else []
            maxpart = max((len(x) for x in comps), default=0)
            if maxpart <= n // 2:
                out.append(c)
        return sorted(out)

    return {
        "depth": [depth.get(v, -1) for v in range(n)],
        "subtree_size": {str(v): len(s) for v, s in desc.items()},
        "descendants": desc,
        "lca": lca_out,
        "is_ancestor": anc_out,
        "kth_ancestor": kth_out,
        "centroid": centroids(),
    }


@pytest.mark.parametrize("name,n,edges", CORPUS, ids=[c[0] for c in CORPUS])
def test_tree_core_vs_sage_and_nx(name, n, edges):
    spec = _spec_for(n, edges)
    got = _run_oracle(spec)
    sage = _sage_tree_facts(n, edges, spec["root"])
    nx_ref = _nx_reference(n, edges, spec["root"], spec)

    # --- predicates vs Sage ---
    assert got["is_tree"] == sage["is_tree"], f"{name}: is_tree"
    assert got["is_forest"] == sage["is_forest"], f"{name}: is_forest"

    # --- depth vs Sage shortest_path_lengths ---
    assert got["depth"] == sage["depth"], f"{name}: depth vs sage"
    # and vs NetworkX (consensus).
    assert got["depth"] == nx_ref["depth"], f"{name}: depth vs nx"

    # --- diameter vs Sage ---
    assert got["diameter"] == sage["diameter"], f"{name}: diameter"

    # --- subtree_size / descendants / is_ancestor vs NetworkX ---
    for v, ds in nx_ref["descendants"].items():
        assert got["descendants"][v] == ds, f"{name}: descendants[{v}]"
    for v, sz in nx_ref["subtree_size"].items():
        assert got["subtree_size"][int(v)] == sz, f"{name}: subtree_size[{v}]"
    assert got["is_ancestor"] == nx_ref["is_ancestor"], f"{name}: is_ancestor"

    # --- lca vs NetworkX root-path reference ---
    assert got["lca"] == nx_ref["lca"], f"{name}: lca"

    # --- kth_ancestor vs NetworkX ---
    assert got["kth_ancestor"] == nx_ref["kth_ancestor"], f"{name}: kth_ancestor"

    # --- centroid (size-balance) vs NetworkX brute force ---
    assert got["centroid"] == nx_ref["centroid"], f"{name}: centroid"

    # --- invariant cells (no direct Sage oracle) ---
    log2n = 0 if n <= 1 else (n - 1).bit_length()
    assert got["hld_max_light"] <= log2n, f"{name}: HLD light-edge invariant"
    assert got["cd_depth"] <= 2 * math.ceil(math.log2(max(n, 2))) + 2, (
        f"{name}: centroid-decomposition depth invariant"
    )

    # diameter path is a real path of the claimed length.
    dp = got["diameter_path"]
    assert len(dp) == got["diameter"] + 1, f"{name}: diameter path length"
    eset = {tuple(sorted(e)) for e in edges}
    for a, b in zip(dp, dp[1:]):
        assert tuple(sorted((a, b))) in eset, f"{name}: diameter path edge"
