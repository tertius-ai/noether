# Contexts and precision

A context specifies the ring or field in which a value lives. It constructs
coefficients and records the data needed to interpret them: a modulus,
defining polynomial, variable names, monomial order, or precision.

## Constructing coefficients

`Z` and `Q` are the integer and rational contexts. Rational coefficients
are exact; use `Fraction` when constructing a nonintegral rational:

```python
from fractions import Fraction

from noether import Q
from noether.commalg import ResidueRing
from noether.linalg import Matrix

A = Matrix.from_rows(Q, [[Fraction(1, 3), 0], [0, 2]])
assert A.det() == Fraction(2, 3)

mod12 = ResidueRing(12)
assert mod12(14) == mod12(2)
```

The residue class `mod12(2)` belongs to ℤ/12, where it has no multiplicative
inverse. Choosing a residue ring rather than a field changes which
operations are defined. In particular, field-based matrix solving requires
a field context; use `solve_exact()` for congruences over a composite modulus.

## Context compatibility

Two finite fields can have the same size and different presentations.
An explicit isomorphism specifies how their generators correspond:

```python
from noether.numthy import FiniteField

source = FiniteField.extension(5, 2)
target = FiniteField.with_modulus(5, [2, 0, 1])
isomorphism = source.isomorphism_to(target)
value = source.element_from_coefficients([3, 2])
assert isomorphism.apply_inverse(isomorphism(value)) == value
isomorphism.verify()
```

Arithmetic rejects incompatible contexts with `ContextMismatchError`.
Reconstructing an equal context is allowed: identity depends on its
mathematical data, rather than the Python object's address. Matrices carry
their coefficient context too; equal shapes alone do not make them compatible.

## Polynomial variables and order

```python
from noether import Q
from noether.commalg import PolynomialRing

ring = PolynomialRing(Q, ["x", "y"], order="lex")
x, y = ring.generators()
assert (x + y)**2 == x**2 + 2*x*y + y**2
```

The variable list fixes exponent-vector coordinates. Monomial order affects
leading terms and Gröbner bases. Both belong to the polynomial context.
A grading additionally assigns degrees to variables. Polynomial grading
does not discard terms.

## Fixed p-adic and q-adic precision

`PadicRing(p, absolute_precision=N)` represents p-adic integers modulo pᴺ.
Choose the cap when constructing the context:

```python
from noether.numthy import PadicRing

integers = PadicRing(5, absolute_precision=6)
assert integers(5**6) == integers.zero()
```

`PadicField` records a fixed absolute cap and its elements report their
known precision. A value's relative precision can decrease during
arithmetic; the context does not silently gain digits. Restriction maps
lower precision. Additional digits require an explicitly justified source.

`QadicRing(residue_field, absolute_precision=N)` represents truncated Witt
vectors over the specified finite field; `QadicField` is the corresponding
field context. A ramified `PadicLocalField` also records its defining
polynomial and a `PadicLocalPrecision` measured in uniformizer powers.
See [Hensel lifting](../examples/10_padic_hensel_lifting.py),
[Teichmüller lifts](../examples/11_qadic_witt_vectors.py), and
[ramified local precision](../examples/31_local_fields_and_precision.py).

## Power-series truncation

A power-series context specifies which monomials are retained:

```python
from noether import Z
from noether.commalg import PowerSeriesRing, SeriesTruncation

series = PowerSeriesRing(
    Z, ["x", "y"], SeriesTruncation.weighted_degree([1, 2], 6)
)
x, y = series.generators()
assert (x**6).is_zero()
assert not (x * y).is_zero()
```

Here xᵃyᵇ is retained exactly when a + 2b < 6. Equality in this truncated
ring compares the retained coefficients; omitted terms have been discarded.
Weights must be positive. Other truncations use total degree or a
per-variable box. In the univariate absolute-precision convention, a bound
N retains exponents strictly below N.

## Ball precision

`RealBallField(bits)` and `ComplexBallField(bits)` construct enclosures at
a specified working precision. Inexact inputs and rounding contribute to
the enclosure width. An overlap test establishes that the enclosures meet;
it does not establish equality of their represented exact values.
See [Ball arithmetic](../examples/02_ball_arithmetic.py).
