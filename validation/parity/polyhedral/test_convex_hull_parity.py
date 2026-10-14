# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Oracle parity for the public exact bounded-polytope facade.

Cross-checks the engine's polytope surface against **Sage** (`Polyhedron`,
cddlib/PPL backend). Every accepted construction direction appears in an
oracle-compared cell:

  V-rep in     convex_hull → f-vector, vertex set, facet count
  H-rep in     Polytope.from_halfspaces on Sage's own inequalities
  combinatoric face_lattice / f_vector (representation-independent invariant)
  predicates   is_vertex / is_facet

Input matrix: dimension ∈ {2,3,4,6,8}; families simplex, cube,
cross-polytope, cyclic, random 0/1, a non-simple (prism) and a
non-simplicial (square pyramid) example; both enumeration directions.

Gated on ``NOETHER_VALIDATE_SAGE=1`` + a Sage binary; never a CI dependency
(runs via ``make validate`` / the node gate on the oracle box). Combinatorial
invariants are compared exactly; vertex *coordinates* are compared as exact
rational sets (canonical); facet coordinates are basis-dependent for a
lower-dimensional hull and are compared only as counts there.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
from fractions import Fraction

import pytest

from noether.geometry import Polytope, convex_hull


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


# ── parity corpus ────────────────────────────────────────────────────
def _simplex(d):
    return [[0] * d] + [[1 if i == j else 0 for j in range(d)] for i in range(d)]


def _cube(d):
    return [list(bits) for bits in itertools.product([0, 1], repeat=d)]


def _cross(d):
    v = []
    for i in range(d):
        e = [0] * d
        e[i] = 1
        v.append(e[:])
        e[i] = -1
        v.append(e[:])
    return v


def _cyclic(n, d):
    # moment curve t ↦ (t, t², …, t^d), t = 1..n
    return [[t**k for k in range(1, d + 1)] for t in range(1, n + 1)]


def _random01(d, m, seed):
    # m distinct random {0,1} vertices (deterministic LCG)
    s = seed
    seen, out = set(), []
    while len(out) < m and len(seen) < (1 << d):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        bits = tuple((s >> i) & 1 for i in range(d))
        if bits not in seen:
            seen.add(bits)
            out.append(list(bits))
    return out


_PYRAMID = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 1, 2]]  # square base + apex
_PRISM = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]  # tri prism
_FRAC_TRI = [["1/2", "0"], ["0", "1/3"], ["-1/2", "-1/3"]]  # exact non-integer coords


def _corpus():
    cases = []

    def add(name, verts, full=True):
        cases.append({"name": name, "vertices": verts, "full": full})

    for d in (2, 3, 4, 6):
        add(f"simplex_{d}", _simplex(d))
    for d in (2, 3, 4, 6, 8):
        add(f"cube_{d}", _cube(d))
    for d in (2, 3, 4, 6, 8):
        add(f"cross_{d}", _cross(d))
    add("cyclic_4_8", _cyclic(8, 4))
    add("cyclic_6_9", _cyclic(9, 6))
    add("cyclic_8_11", _cyclic(11, 8))
    add("random01_5a", _random01(5, 12, 7))
    add("random01_6a", _random01(6, 16, 13))
    add("pyramid_3", _PYRAMID)
    add("prism_3", _PRISM)
    add("frac_triangle", _FRAC_TRI)
    # a 2-square embedded in R³ — a lower-dimensional hull
    add("square_in_3d", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], full=False)
    return cases


CORPUS = _corpus()


def _to_fraction_rows(rows):
    return [[Fraction(str(c)) for c in r] for r in rows]


def _vset(rows):
    """A representation-free vertex set: frozenset of Fraction tuples."""
    return frozenset(tuple(Fraction(str(c)) for c in r) for r in rows)


# ── the Sage oracle (one subprocess for the whole corpus) ───────────
_SAGE_TEMPLATE = r"""
import json, sys
from sage.all import Polyhedron, QQ

corpus = json.loads(sys.argv[1])
out = {}
for case in corpus:
    V = [[QQ(c) for c in v] for v in case["vertices"]]
    P = Polyhedron(vertices=V, base_ring=QQ)
    # inequalities: Sage stores b + a·x >= 0 as [b, a1..ad]; Noether wants
    # <c,x> <= e, i.e. c = -a, e = b.
    ineqs = []
    for row in P.inequalities_list():
        b = row[0]
        a = row[1:]
        ineqs.append({"c": [str(-ai) for ai in a], "e": str(b)})
    verts = sorted([[str(c) for c in v] for v in P.vertices_list()])
    out[case["name"]] = {
        "fvector": [int(x) for x in P.f_vector()],
        "vertices": verts,
        "n_facets": int(P.n_facets()),
        "n_equations": len(P.equations_list()),
        "ineqs": ineqs,
    }
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def oracle():
    corpus_json = json.dumps(CORPUS)
    r = subprocess.run(
        [SAGE, "-python", "-c", _SAGE_TEMPLATE, corpus_json],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if r.returncode != 0:
        raise RuntimeError(f"sage oracle failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _noether_hull(case):
    return convex_hull(_to_fraction_rows(case["vertices"]))


# ── tests ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", CORPUS, ids=[c["name"] for c in CORPUS])
def test_v_rep_parity(case, oracle):
    """convex_hull: f-vector, vertex set, and facet count vs Sage."""
    ref = oracle[case["name"]]
    P = _noether_hull(case)
    assert P.f_vector == ref["fvector"], f"{case['name']} f-vector"
    assert _vset(P.vertices) == _vset(ref["vertices"]), f"{case['name']} vertices"
    assert len(P.vertices) == len(ref["vertices"]), f"{case['name']} n_vertices"
    # n_facets = f_vector[-2] (Sage convention); for a full-dim hull this is
    # also len(inequalities).
    assert len(P.vertex_facet_incidence) == ref["fvector"][-2], f"{case['name']} n_facets"


@pytest.mark.parametrize(
    "case",
    [c for c in CORPUS if c["full"]],
    ids=[c["name"] for c in CORPUS if c["full"]],
)
def test_h_to_v_parity(case, oracle):
    """Polytope.from_halfspaces on Sage's own exact H-representation."""
    ref = oracle[case["name"]]
    if ref["n_equations"] != 0:
        pytest.skip("lower-dimensional H-rep with equations")
    a = _to_fraction_rows([iq["c"] for iq in ref["ineqs"]])
    b = [Fraction(iq["e"]) for iq in ref["ineqs"]]
    polytope = Polytope.from_halfspaces(a, b)
    assert _vset(polytope.vertices) == _vset(ref["vertices"]), f"{case['name']} H→V vertices"


def test_predicates_and_face_lattice(oracle):
    """is_vertex / is_facet / face_lattice on a hand-checked polytope."""
    # unit cube in R³
    P = convex_hull(_cube(3))
    assert P.is_vertex([0, 0, 0])
    assert P.is_vertex([1, 1, 1])  # a cube vertex
    assert not P.is_vertex([1, 1, 2])  # outside
    assert not P.is_vertex([1, 1])  # wrong ambient dim
    # a facet of the cube: x0 <= 1  → normal (1,0,0), rhs 1
    assert P.is_facet([1, 0, 0], 1)
    assert not P.is_facet([1, 0, 0], 2)  # not tight
    # face lattice: counts by dimension must match the f-vector.
    faces = P.face_lattice()
    counts = {}
    for dim, _ in faces:
        counts[dim] = counts.get(dim, 0) + 1
    fv = P.f_vector
    for dim, c in counts.items():
        assert c == fv[dim + 1], f"face-lattice dim {dim}"
    # covers form a graded Hasse diagram (each cover lifts dimension by 1).
    covers = P.face_lattice_covers()
    assert len(covers) > 0
    for lo, hi in covers:
        assert faces[hi][0] == faces[lo][0] + 1
