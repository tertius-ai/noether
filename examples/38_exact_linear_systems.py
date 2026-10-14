# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact affine systems and canonical submodules over Z and Z/nZ."""

from __future__ import annotations

from noether import IntegerRing, Z
from noether.commalg import ResidueRing
from noether.homalg import FreeModule, ModuleElement, ModuleHomomorphism, Submodule
from noether.linalg import Matrix, Vector

operator = Matrix.from_rows(Z, [[2, 4, 0], [0, 0, 3]])
right_hand_side = Vector.from_values(Z, [6, 9])
integer_result = operator.solve_exact(right_hand_side)
assert integer_result.is_consistent
integer_solutions = integer_result.solutions
assert integer_solutions is not None
assert operator @ integer_solutions.particular_solution == right_hand_side
assert operator @ integer_solutions.homogeneous_generators == Matrix.zeros(Z, (2, 1))
integer_result.verify()

# Empty systems retain a replayable divisibility obstruction.
impossible = Matrix.from_rows(Z, [[6]]).solve_exact(Vector.from_values(Z, [9]))
assert not impossible.is_consistent
assert impossible.obstruction is not None
assert impossible.obstruction.left_gcd == 6
assert abs(impossible.obstruction.combined_rhs) == 9
impossible.obstruction.verify()

# Composite moduli are not treated as fields.  The result is an affine module,
# including all homogeneous torsion directions.
z12 = ResidueRing(12)
congruence = Matrix.from_rows(z12, [[8]]).solve_exact(Vector.from_values(z12, [4]))
assert congruence.is_consistent
congruence_solutions = congruence.solutions
assert congruence_solutions is not None
assert {
    (int(congruence_solutions.particular_solution[0]) + 3 * offset) % 12 for offset in range(4)
} == {
    2,
    5,
    8,
    11,
}
congruence.verify()

# Submodules retain a canonical normal form, so equality and lattice operations
# compare mathematics rather than the supplied generating lists.
ambient = FreeModule(Z, 2)


def element(x: int, y: int) -> ModuleElement[IntegerRing]:
    return ModuleElement(ambient, Vector.from_values(Z, [x, y]))


lattice = Submodule.from_generators(ambient, [element(2, 0), element(0, 3)])
assert lattice.finite_index() == 6
witness = lattice.membership_witness(element(6, 9))
assert witness is not None
witness.verify()
assert lattice.saturation().finite_index() == 1

homomorphism = ModuleHomomorphism(
    FreeModule(Z, 3),
    ambient,
    operator,
)
kernel_result = homomorphism.kernel()
image_result = homomorphism.image()
kernel = kernel_result.submodule
image = image_result.submodule
assert kernel.generator_count == 1
assert image.contains(element(2, 0))
kernel_result.verify()
image_result.verify()

print(
    "exact systems: integer affine rank 1; four Z/12Z solutions; "
    "index-6 lattice saturated and verified"
)
