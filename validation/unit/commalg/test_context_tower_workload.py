# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The fixed 0.1 coefficient contexts compose without losing identity."""

from noether.commalg import (
    CrtDecomposition,
    Polynomial,
    PolynomialRing,
    PowerSeries,
    PowerSeriesRing,
    RationalFunction,
    RationalFunctionField,
    ResidueClass,
    ResidueRing,
    SeriesTruncation,
    WeightedGrading,
)
from noether.numthy import (
    FiniteField,
    FiniteFieldElement,
    FiniteFieldIsomorphism,
    PadicElement,
    PadicField,
    PadicRing,
    QadicElement,
    QadicField,
    QadicRing,
)


def test_fixed_context_tower_round_trips_without_erasing_parents() -> None:
    f3 = FiniteField.prime(3)
    f9 = FiniteField.extension(3, 2)
    z_mod_81 = ResidueRing(3**4)
    z3 = PadicRing(3, absolute_precision=6)
    q3 = PadicField(3, absolute_precision=6)
    w9 = QadicRing(f9, absolute_precision=6)
    q9 = QadicField(f9, absolute_precision=6)
    polynomials = PolynomialRing(
        f9,
        ["x", "y"],
        grading=WeightedGrading([1, 2]),
    )
    functions = RationalFunctionField(f9, "c")
    series = PowerSeriesRing(
        w9,
        ["x", "y", "z"],
        SeriesTruncation.weighted_degree([1, 2, 3], 6),
    )

    contexts = [
        (FiniteField, f3),
        (FiniteField, f9),
        (ResidueRing, z_mod_81),
        (PadicRing, z3),
        (PadicField, q3),
        (QadicRing, w9),
        (QadicField, q9),
        (PolynomialRing, polynomials),
        (RationalFunctionField, functions),
        (PowerSeriesRing, series),
    ]
    for context_type, context in contexts:
        assert context_type.from_json(context.to_json()) == context

    a = f9.generator()
    x, y = polynomials.generators()
    c = functions.generator()
    sx, sy, sz = series.generators()
    values = [
        (FiniteFieldElement, f3(2)),
        (FiniteFieldElement, a + f9.one()),
        (ResidueClass, z_mod_81(-7)),
        (PadicElement, z3(-1)),
        (PadicElement, q3.inverse(q3(3))),
        (QadicElement, w9.generator() + w9.one()),
        (QadicElement, q9.inverse(q9(3))),
        (Polynomial, x**2 + a * y),
        (RationalFunction, (c + a) / (c + 1)),
        (PowerSeries, 1 + sx + sy + sx * sy * sy + sz),
    ]
    for value_type, value in values:
        assert value_type.from_json(value.to_json()) == value

    crt = z_mod_81.crt()
    components = crt.project(z_mod_81(-7))
    assert [factor.modulus for factor in crt.factors] == [81]
    assert crt.reconstruct(components) == z_mod_81(-7)
    assert CrtDecomposition.from_json(crt.to_json()) == crt


def test_residue_frobenius_and_weighted_precision_identities() -> None:
    f9 = FiniteField.extension(3, 2)
    a = f9.generator() + f9.one()
    assert a.frobenius(2) == a
    assert a.trace().frobenius(1) == a.trace()
    assert a.norm().frobenius(1) == a.norm()

    z3 = PadicRing(3, absolute_precision=6)
    residue = z3.residue_field(2)
    lift = z3.teichmuller(residue)
    assert lift**3 == lift
    assert lift.residue() == residue

    hensel_polynomials = PolynomialRing(z3, ["t"])
    t = hensel_polynomials.generator(0)
    hensel_root = z3.hensel_lift(t**2 - 7, z3.residue_field(1))
    assert (t**2 - 7).evaluate([hensel_root]) == z3.zero()
    assert hensel_root.residue() == z3.residue_field(1)

    w9 = QadicRing(f9, absolute_precision=6)
    extension_residue = f9.generator() + f9.one()
    extension_lift = w9.teichmuller(extension_residue)
    assert extension_lift.residue() == extension_residue
    assert extension_lift.frobenius(2) == extension_lift
    assert w9.one().trace() == w9.coefficient_ring(2)
    assert w9(3).norm() == w9.coefficient_ring(9)

    q3 = PadicField(3, absolute_precision=6)
    q9 = QadicField(f9, absolute_precision=6)
    assert q3.inverse(q3(3)).valuation().value == -1
    assert q9.inverse(q9(3)).valuation().value == -1

    truncation = SeriesTruncation.weighted_degree([1, 2, 3], 6)
    retained = [
        (i, j, k)
        for i in range(6)
        for j in range(3)
        for k in range(2)
        if truncation.contains_exponent([i, j, k])
    ]
    assert len(retained) == 16
    assert (1, 2, 0) in retained
    assert (0, 3, 0) not in retained

    series = PowerSeriesRing(w9, ["x", "y", "z"], truncation)
    x, y, z = series.generators()
    value = 1 + x + y + x * y * y + z
    assert value.nonzero_term_count == 5
    assert value.ring == series
    assert (
        value.restrict_truncation(SeriesTruncation.weighted_degree([1, 2, 3], 4)).nonzero_term_count
        == 4
    )


def test_distinct_finite_field_presentations_use_an_explicit_checked_map() -> None:
    source = FiniteField.extension(5, 2)
    target = FiniteField.with_modulus(5, [2, 0, 1])
    isomorphism = source.isomorphism_to(target)

    assert source != target
    assert isomorphism.source == source
    assert isomorphism.target == target
    isomorphism.verify()

    value = source.element_from_coefficients([3, 2])
    image = isomorphism(value)
    assert image.field == target
    assert isomorphism.apply_inverse(image) == value
    assert FiniteFieldIsomorphism.from_json(isomorphism.to_json()) == isomorphism
