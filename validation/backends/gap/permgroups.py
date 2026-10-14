# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""GAP oracle for the `bsgs` topic — permutation groups.

A thin wrapper around `gap` in batch mode that, for an input generator
list (image-vector form), computes the invariants we cross-
check the Noether kernel against:

* `Size(G)` — group order
* `OrbitDomain(G, p)` — the orbit of `p` (sorted, so the comparison
  is order-independent)
* sorted multiset of base-orbit sizes along a stabilizer chain seeded
  at `p`
* `Size(Stabilizer(G, [p_1, p_2, ...]))` — pointwise stabilizer order
* membership verdicts (`PermList(...) in G`) for a list of probe
  permutations

The Noether image-vector convention (`images[i]` is the image of
`i ∈ 0..degree`) is translated to GAP's 1-based image lists via
`PermList([images[0]+1, images[1]+1, ...])`.
"""

from __future__ import annotations

import re
from typing import Iterable

from validation.backends.gap._run import GapError, run_gap


def _gap_perm(images: list[int]) -> str:
    """Render an image-vector permutation as a GAP `PermList(...)` call.

    GAP is 1-based; Noether is 0-based. `images` is the 0-based image
    vector (length `degree`); we render `[images[0]+1, ..., images[deg-1]+1]`.
    """
    return "PermList([" + ",".join(str(int(x) + 1) for x in images) + "])"


def _gap_perm_list(perms: list[list[int]]) -> str:
    return "[" + ",".join(_gap_perm(p) for p in perms) + "]"


def _gap_int_list(xs: Iterable[int]) -> str:
    return "[" + ",".join(str(int(x) + 1) for x in xs) + "]"


def _unwrap(out: str) -> str:
    """GAP wraps lines at column ~80 with a trailing `\\` and a
    leading-whitespace continuation. Stitch those back together so a
    line-anchored regex matches a logical record."""
    return re.sub(r"\\\n\s*", "", out)


def _parse_int(out: str, key: str) -> int:
    out = _unwrap(out)
    m = re.search(rf"^{re.escape(key)}: (-?\d+)$", out, re.MULTILINE)
    if m is None:
        raise GapError(f"could not parse {key} from GAP output:\n{out}")
    return int(m.group(1))


def _parse_int_list(out: str, key: str) -> list[int]:
    out = _unwrap(out)
    m = re.search(rf"^{re.escape(key)}: \[ *([0-9, ]*) *\]$", out, re.MULTILINE)
    if m is None:
        raise GapError(f"could not parse {key} from GAP output:\n{out}")
    body = m.group(1).strip()
    if not body:
        return []
    return [int(x) for x in body.split(",")]


def _parse_bool_list(out: str, key: str) -> list[bool]:
    out = _unwrap(out)
    m = re.search(rf"^{re.escape(key)}: \[ *([a-z, ]*) *\]$", out, re.MULTILINE)
    if m is None:
        raise GapError(f"could not parse {key} from GAP output:\n{out}")
    body = m.group(1).strip()
    if not body:
        return []
    return [tok.strip() == "true" for tok in body.split(",")]


def gap_perm_group_report(
    *,
    degree: int,
    generators: list[list[int]],
    orbit_points: list[int],
    pointwise_stab: list[int],
    probes: list[list[int]],
    chain_orbit_seed: list[int],
) -> dict:
    """Run GAP on a permutation group built from `generators`.

    Returns a dict with the invariants the parity harness compares.

    * `degree` — the underlying domain `1..degree` (GAP) / `0..degree`
      (Noether); the same integer.
    * `generators` — 0-based image vectors.
    * `orbit_points` — for each `p`, return the sorted 0-based orbit
      of `p`.
    * `pointwise_stab` — 0-based point list; return the order of
      `Stab_G(pointwise_stab)`.
    * `probes` — 0-based image vectors of probe permutations; return
      `[g in G for g in probes]`.
    * `chain_orbit_seed` — 0-based base seed for a stabilizer-chain
      orbit-size signature (a sorted multiset of per-level orbit
      lengths).
    """
    gens_gap = _gap_perm_list(generators)
    parts: list[str] = []
    parts.append(f"deg := {int(degree)};")
    parts.append("Gens := " + gens_gap + ";;")
    # Drop the trivial-only generator case: GAP rejects [()] for
    # `Group` but accepts `Group([(),()])` happily; safer to fall back
    # to `TrivialGroup(IsPermGroup)` if every generator is the identity.
    parts.append(
        "if ForAll(Gens, g -> g = ()) then G := SymmetricGroup(deg)^Identity(SymmetricGroup(deg));"
        " G := TrivialSubgroup(SymmetricGroup(deg));"
        " else G := Group(Gens); fi;"
    )
    parts.append('Print("ORDER: ", Size(G), "\\n");')

    # Sorted orbits for each requested point.
    for i, p in enumerate(orbit_points):
        parts.append(
            f'Print("ORBIT_{i}: ", "[", JoinStringsWithSeparator('
            f'List(SortedList(Orbit(G, {int(p) + 1})), x -> String(x - 1)), ","), "]", "\\n");'
        )

    # Pointwise stabilizer order.
    if pointwise_stab:
        ps = _gap_int_list(pointwise_stab)
        parts.append(f"Stb := Stabilizer(G, {ps}, OnTuples);;")
        parts.append('Print("PSTAB_ORDER: ", Size(Stb), "\\n");')
    else:
        parts.append('Print("PSTAB_ORDER: ", Size(G), "\\n");')

    # Probe membership: each probe is a 0-based image vector → PermList
    # → `in G`. GAP prints booleans as `true` / `false`.
    if probes:
        probes_gap = _gap_perm_list(probes)
        parts.append("Probes := " + probes_gap + ";;")
        parts.append(
            'Print("MEMBERSHIP: ", "[", JoinStringsWithSeparator('
            'List(Probes, g -> String(g in G)), ","), "]", "\\n");'
        )
    else:
        parts.append('Print("MEMBERSHIP: [ ]\\n");')

    # Stabilizer chain orbit sizes (sorted) seeded at `chain_orbit_seed`.
    # We follow the base in given order, computing |Orb_{G_i}(base_i)|
    # at each level via Stabilizer(.., OnTuples).
    if chain_orbit_seed:
        seed_gap = _gap_int_list(chain_orbit_seed)
        parts.append("ChainSizes := [];;")
        parts.append("Acc := G;;")
        parts.append("BaseSeq := " + seed_gap + ";;")
        parts.append(
            "for i in [1..Length(BaseSeq)] do "
            "Add(ChainSizes, Length(Orbit(Acc, BaseSeq[i]))); "
            "Acc := Stabilizer(Acc, BaseSeq[i]); od;"
        )
        # Sorted multiset to make this base-ordering-invariant.
        parts.append(
            'Print("CHAIN_ORBIT_SIZES: ", "[", JoinStringsWithSeparator('
            'List(SortedList(ChainSizes), x -> String(x)), ","), "]", "\\n");'
        )
    else:
        parts.append('Print("CHAIN_ORBIT_SIZES: [ ]\\n");')

    parts.append("QUIT;")
    script = "\n".join(parts) + "\n"
    out = run_gap(script)

    report = {
        "order": _parse_int(out, "ORDER"),
        "orbits": [tuple(_parse_int_list(out, f"ORBIT_{i}")) for i in range(len(orbit_points))],
        "pstab_order": _parse_int(out, "PSTAB_ORDER"),
        "membership": tuple(_parse_bool_list(out, "MEMBERSHIP")),
        "chain_orbit_sizes": tuple(_parse_int_list(out, "CHAIN_ORBIT_SIZES")),
    }
    return report
