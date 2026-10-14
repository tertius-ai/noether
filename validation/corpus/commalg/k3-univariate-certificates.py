# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify exact univariate certificates from a K3-surface computation."""

from fractions import Fraction

from noether import Q, Z
from noether.commalg import PolynomialRing
from noether.numthy import FiniteField, cyclotomic_polynomial


def verify_weil_factorization() -> None:
    ring = PolynomialRing(Q, ["x"], order="lex")
    numerators = [
        31,
        -12,
        15,
        -6,
        -3,
        -6,
        -5,
        5,
        -13,
        15,
        -22,
        15,
        -13,
        5,
        -5,
        -6,
        -3,
        -6,
        15,
        -12,
        31,
    ]
    degree_twenty = ring.from_terms(
        ([exponent], Fraction(numerator, 31)) for exponent, numerator in enumerate(numerators)
    )
    phi_one = cyclotomic_polynomial(1).base_change(Z.rational_embedding())
    normalized_weil = phi_one**2 * degree_twenty
    factorization = normalized_weil.factor()
    factorization.verify()
    assert factorization.coefficient == 1
    assert set(factorization.factors) == {(phi_one, 2), (degree_twenty, 1)}

    for index in [25, 33, 44, 50, 66]:
        candidate = cyclotomic_polynomial(index).base_change(Z.rational_embedding())
        assert degree_twenty != candidate


def verify_restricted_sextics() -> None:
    cases = [
        (5, [4, 2, 3, 1, 1]),
        (31, [20, 17, 2, 27, 22, 10, 4]),
    ]
    for prime, coefficients in cases:
        field = FiniteField.prime(prime)
        ring = PolynomialRing(field, ["t"], order="lex")
        restricted = ring.from_terms(
            ([exponent], field(coefficient)) for exponent, coefficient in enumerate(coefficients)
        )
        assert restricted.is_nth_power(2)
        root = restricted.nth_root(2)
        assert root is not None
        assert root**2 == restricted
        assert root.degree() is not None and root.degree() <= 3


def verify_critical_value_root() -> None:
    coefficients = [
        3360562185644360160973024557760060102333744225,
        72583254065809636635306332337703824366188434270,
        726654657703922408940557551155032090684638933826,
        4491703719438691166049926993010365150003754711011,
        19256353004968473541580950264495768027780300745039,
        60962873077723427409680718610171857266200338167358,
        148201393034895700713856853648997648220419859045614,
        284110776451050852671039168867589647389115131321640,
        437803897038398722059166553513985177579371161559422,
        549949732424928206881207552937639747979608663294517,
        568814322958527604958344109338564919774781998504501,
        487510238853906640383319739836454123296419603393994,
        347120875264540046734539521147909916728268867388475,
        205001911678589276856746882042973886033642846174586,
        99795207219245456392194890309892042934811552651358,
        39637318624461089569909394131301819370491354575414,
        12746752079295662883612929235826851436347673895036,
        3411467117927526957743561248737057757569536641175,
        897931764752726405097675671836017715602757282832,
        313702111241376644531747319308739183459079705481,
        134635992963754717236436477478853663889787847041,
        51203458867071515178816535793913929841295545523,
        14781904815184226503773321261056140343696882688,
        3030216345151105843803927021876499585901617152,
        402365467566860107202765507308898559483445248,
        27003416146933923083721127498966008518934528,
    ]
    ring = PolynomialRing(Z, ["t"], order="lex")
    critical = ring.from_terms(
        ([exponent], coefficient) for exponent, coefficient in enumerate(coefficients)
    )
    roots = critical.real_roots()
    assert len(roots) == 1
    root = roots[0]
    root.verify()
    assert root.sign() == -1
    assert root.refine(32).upper < 0
    assert not root.ball(192).contains_zero()


def main() -> None:
    verify_weil_factorization()
    verify_restricted_sextics()
    verify_critical_value_root()
    print("verified Weil, finite-field square-root, and real-root certificates")


if __name__ == "__main__":
    main()
