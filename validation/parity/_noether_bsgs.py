# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Noether oracle for the `bsgs` parity topic.

Calls the `noether.groups.PermutationGroup` PyO3 projection directly —
no subprocess, no JSON. The `noether_report` function returns the
same dict shape as `validation/backends/gap/permgroups.py`'s
`gap_perm_group_report`, so the two are line-for-line comparable.
"""

from __future__ import annotations

from typing import Any


def _permgroup_module():
    """Lazily import the public group projection."""
    from noether import groups

    return groups


def noether_report(
    *,
    degree: int,
    generators: list[list[int]],
    orbit_points: list[int],
    pointwise_stab: list[int],
    probes: list[list[int]],
    chain_orbit_seed: list[int],
    random_seeds: list[int],
) -> dict[str, Any]:
    """Build the accepted invariant report on the Noether side.

    Mirrors `gap_perm_group_report` field-for-field on the shared
    keys (`order`, `orbits`, `pstab_order`, `membership`,
    `chain_orbit_sizes`) and adds Noether-only sanity invariants the
    parity harness uses to verify functions without an equivalent GAP
    column for.
    """
    ne = _permgroup_module()
    perms = [ne.Permutation.from_images(g) for g in generators]
    G = ne.PermutationGroup.from_generators(degree, perms)

    order = int(G.order)

    # Orbits — sorted tuples for comparability.
    orbits: list[tuple[int, ...]] = []
    for p in orbit_points:
        ob = G.orbit(p)
        orbits.append(tuple(sorted(ob)))

    # Pointwise stabilizer order — covers `pointwise_stabilizer`.
    if pointwise_stab:
        stab = G.pointwise_stabilizer(pointwise_stab)
        pstab_order = int(stab.order)
    else:
        pstab_order = order

    # Membership for each probe — covers `element_membership_test`.
    membership: list[bool] = []
    for img in probes:
        pp = ne.Permutation.from_images(img)
        membership.append(G.contains(pp))

    # Stabilizer-chain orbit sizes along the seed, then sorted to
    # make the invariant base-ordering-independent.
    chain_sizes: list[int] = []
    acc = G
    for b in chain_orbit_seed:
        chain_sizes.append(len(acc.orbit(b)))
        acc = acc.pointwise_stabilizer([b])
    chain_sizes_sorted = tuple(sorted(chain_sizes))

    # ── Noether-only invariants for functions without a parallel GAP
    # column in this report layout. ──────────────────────────────────

    # Schreier-generators: build a fresh PermutationGroup from the Schreier
    # gens of Stab_G(0) and check the order matches |Stab_G(0)|. This
    # is Schreier's lemma — a kernel where `schreier_generators` is
    # broken would fail this.
    schreier_lemma_holds: bool = True
    try:
        if degree >= 1:
            sg = G.schreier_generators(0)
            sg_perms = [s.perm for s in sg]
            if sg_perms:
                sg_group = ne.PermutationGroup.from_generators(degree, sg_perms)
                # |Stab_G(0)|
                expected = int(G.pointwise_stabilizer([0]).order)
                schreier_lemma_holds = int(sg_group.order) == expected
    except Exception:  # noqa: BLE001
        schreier_lemma_holds = False

    # iterate_elements: the count must equal |G| when feasible. We
    # cap iteration here — the harness's `_to_canonical` strips
    # Noether-only fields, so this fact stays on the Noether side.
    if order <= 1000:
        iterated = sum(1 for _ in G.iterate_elements())
    else:
        iterated = None

    # uniform_random_element: every seeded draw must lie in G.
    random_members = []
    for s in random_seeds:
        r = G.uniform_random_element(s)
        random_members.append(G.contains(r.element))

    # schreier_tree: every non-root point has a parent in the orbit
    # whose generator-witness reproduces it. Quick structural sanity.
    schreier_tree_ok: bool = True
    try:
        if degree >= 1:
            tree = G.schreier_tree(0)
            gens_used = tree.generators_used
            for p in tree.orbit:
                if p == tree.root:
                    continue
                parent_info = tree.pred[p]
                if parent_info is None:
                    schreier_tree_ok = False
                    break
                parent, gi = parent_info
                if gens_used[gi].apply(parent) != p:
                    schreier_tree_ok = False
                    break
    except Exception:  # noqa: BLE001
        schreier_tree_ok = False

    # transversal: every coset rep maps the base to its orbit point.
    transversal_ok: bool = True
    try:
        if degree >= 1:
            t = G.transversal(0)
            for i, p in enumerate(t.orbit):
                if t.reps[i].apply(t.base) != p:
                    transversal_ok = False
                    break
    except Exception:  # noqa: BLE001
        transversal_ok = False

    # base_and_strong_generating_set: rebuilding the group from the
    # SGS alone must reproduce the order.
    bsgs_round_trip: bool = True
    try:
        (_, sgs) = ne.PermutationGroup.base_and_strong_generating_set(degree, perms)
        if sgs:
            rebuilt = ne.PermutationGroup.from_generators(degree, sgs)
            bsgs_round_trip = int(rebuilt.order) == order
    except Exception:  # noqa: BLE001
        bsgs_round_trip = False

    return {
        # Shared with GAP report.
        "order": order,
        "orbits": orbits,
        "pstab_order": pstab_order,
        "membership": tuple(membership),
        "chain_orbit_sizes": chain_sizes_sorted,
        # Noether-only sanity (these don't go into the canonical
        # comparison but the caller can assert on
        # them).
        "schreier_lemma_holds": schreier_lemma_holds,
        "iterated_count": iterated,
        "random_members": tuple(random_members),
        "schreier_tree_ok": schreier_tree_ok,
        "transversal_ok": transversal_ok,
        "bsgs_round_trip": bsgs_round_trip,
    }
