# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import json
from fractions import Fraction

import pytest

from noether import (
    BezoutIdentity,
    ContextMismatchError,
    IntegerRing,
    NoetherError,
    Q,
    RationalField,
    VerificationError,
    Z,
)


def test_integer_extended_gcd_is_exact_normalized_and_checked() -> None:
    identity = Z.extended_gcd(30, 18)

    assert isinstance(identity, BezoutIdentity)
    assert identity.type_id == "noether/root/BezoutIdentity@1"
    assert identity.ring == Z
    assert identity.gcd == Z.gcd(30, 18) == 6
    assert identity.coefficient_a * 30 + identity.coefficient_b * 18 == 6
    assert identity.verify(Z, 30, 18) is None
    assert "ring=Z" in repr(identity)
    assert identity._tertius_summary_() == {
        "type": "noether/root/BezoutIdentity@1",
        "ring": "Z",
        "gcd": 6,
    }


def test_rational_field_uses_the_canonical_unit_gcd() -> None:
    left = Fraction(2, 3)
    right = Fraction(5, 7)
    identity = Q.extended_gcd(left, right)

    assert identity.ring == Q
    assert identity.gcd == Q.gcd(left, right) == 1
    assert identity.coefficient_a * left + identity.coefficient_b * right == 1
    assert identity.verify(Q, left, right) is None
    assert BezoutIdentity.from_json(identity.to_json()) == identity
    assert Q.gcd(0, 0) == 0
    assert Q.extended_gcd(0, 0).verify(Q, 0, 0) is None


def test_bezout_json_round_trips_and_rejects_wrong_identity() -> None:
    identity = Z.extended_gcd(-(1 << 20_000) + 17, 15)
    encoded = identity.to_json()
    restored = BezoutIdentity.from_json(encoded)

    assert restored == identity
    assert restored.verify(Z, -(1 << 20_000) + 17, 15) is None

    malformed = json.loads(encoded)
    malformed["gcd"] = "7"
    false_identity = BezoutIdentity.from_json(json.dumps(malformed))
    with pytest.raises(VerificationError, match="canonical normalized gcd"):
        false_identity.verify(Z, -(1 << 20_000) + 17, 15)

    malformed["type"] = "not/noether"
    with pytest.raises(NoetherError, match="expected type"):
        BezoutIdentity.from_json(json.dumps(malformed))


def test_bezout_verification_rejects_the_wrong_context() -> None:
    with pytest.raises(ContextMismatchError, match="IntegerRing"):
        Z.extended_gcd(30, 18).verify(Q, 30, 18)


def test_singleton_coefficient_contexts_have_typed_json() -> None:
    assert IntegerRing.type_id == "noether/root/IntegerRing@1"
    assert RationalField.type_id == "noether/root/RationalField@1"
    assert json.loads(Z.to_json()) == {"type": IntegerRing.type_id}
    assert json.loads(Q.to_json()) == {"type": RationalField.type_id}
    assert IntegerRing.from_json(Z.to_json()) == Z
    assert RationalField.from_json(Q.to_json()) == Q

    with pytest.raises(NoetherError, match="expected type"):
        IntegerRing.from_json(Q.to_json())
