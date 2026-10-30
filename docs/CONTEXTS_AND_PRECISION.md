# Contexts and precision

A Noether context is a mathematical parent value: it records the information
that determines coefficient identity and constructs elements belonging to that
identity. Elements carry or reference their context. Equal-looking
coefficients from incompatible contexts do not combine implicitly.

Examples include `Z`, `Q`, `FiniteField`, `ResidueRing`, `PadicRing`,
`QadicRing`, `NumberField`, `PolynomialRing<R>`, `PowerSeriesRing<R>`, and
real/complex ball fields. A finite field context records its characteristic
and presentation; a number field records its defining polynomial; a
polynomial context records variables and monomial order.

## Fixed precision

The local and series workhorses are bounded in advance:

- a `PadicRing` with prime `p` and absolute precision `N` represents integers
  modulo `p^N`;
- a `PadicField` records the same fixed absolute cap and reports each value's
  current relative precision;
- a `QadicRing` is constructed from an explicit finite residue-field context
  and represents its Witt vectors at a fixed absolute precision;
- a `QadicField` is the corresponding fraction field;
- `PowerSeriesRing<R>` uses an explicit finite truncation region.

The fixed cap bounds storage and lets callers make memory guarantees. Loss of
relative p-adic precision is reported by the value; it does not silently
increase the cap. Refinement creates a value in an explicitly refined context.

Hyperelliptic Frobenius allocates one fixed Witt-ring context before
reduction. A proved logarithmic bound on the omitted binomial tail fixes the
truncation. The working cap additionally reserves the sum of all p-adic
valuations of the vertical-reduction divisors. A common denominator is retained
through the reduction; exact uniformizer division then explicitly lowers the
cap to the requested surviving precision. No arithmetic loop refines a
context or invents missing digits. The evidence records and replays these
choices, including small-characteristic reductions.

## Series truncation and grading

Univariate absolute precision `N` retains exponents below `N`. Multivariate
series can use total degree, a per-variable box, or positive weights. With
weights `(1, 2, 3)` and bound `6`, a monomial `x^a y^b z^c` is retained exactly
when `a + 2b + 3c < 6`. Every variable weight must be positive.

Polynomial rings may also carry a grading, but polynomial values are not
truncated by it. The grading supports degree-sensitive mathematics; series
truncation determines which coefficients are physically retained.

Supported univariate fixed-truncation series over exact FLINT contexts keep
sticky native polynomial storage. Weighted multivariate and arbitrary sparse
cases retain the generic exact representation. Both implement the same public
`PowerSeries<R>` operations.

## Matrices over contexts

`Matrix<R>` is parameterized by context, not by a backend element wrapper.
Generic algorithms use the context's algebraic traits. Construction selects
persistent whole-object storage for supported FLINT or kernel regimes, so
ordinary arithmetic avoids packing and unpacking. A generic path remains the
correctness floor for every specialized route.

This distinction is important for extension fields: a matrix over one `F_9`
presentation is not interchangeable with a matrix over another presentation.
An explicit checked `FiniteFieldIsomorphism` maps elements between them.
