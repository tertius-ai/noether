# groebner-kernel

Operation-private Gröbner kernels consumed by `noether::commalg` and
`noether::homalg`. The crate owns fixed polynomial and free-module F4/signature
and integral strong-basis implementations. Its rational
lane uses checked prime images, support learning, CRT/rational reconstruction,
and exact characteristic-zero verification, with direct exact F4 as the
fallback. Ordinary F4 avoids constructing signature multipliers that only the
signature lane consumes. Pair installation likewise defers signature products
until coefficient-independent criteria have retained a compact candidate. It
records exact construction evidence; signature mode and modular reconstruction
receive an independent all-critical-pairs sweep.
The word-prime F4 reducer splices its matrix into sparse known-pivot `A|B` and
dense new-pivot `C|D` blocks, retaining only truncated dense right tails. Its
operation-private sparse rows use checked 32-bit column coordinates matching
the monomial arena's identifier space; native matrix adapters widen only at
their boundaries. Its
exact, dependency-free `C -> D` projection uses crossover-based Rayon dispatch
while preserving input row order. Its sparse upper-pivot updates use an
overflow-proved, eight-way-unrolled lazy multiply-add lane when the actual
modulus and block width permit it, with a separately tuned four-way bounded
delayed-correction lane otherwise.
The dependent `D` echelon phase uses deterministic pivots and cache-sized
delayed-arithmetic tail kernels.
Rational modular runs learn private coefficient-independent row instructions
from the first prime, replay them only while supports and new leading terms
match, and fall back to ordinary F4 for an incompatible prime image.
The higher-rank module lane uses the same packed monomial arena, retains exact
nested Schreyer frames from `module-core`, and batches compatible pairs by
degree. Consumers that need generator coordinates carry them through an
augmented F4 matrix. Basis-only Hilbert and Betti playbooks use a separate
unaugmented reduction: exact row construction keeps its output in the input
span, while a final independent sweep checks reducedness, input containment,
and every critical pair. It therefore avoids allocating and reducing
transformation columns that those playbooks cannot consume.
Its first-syzygy lane assembles all equal-LCM-degree Schreyer pairs into the
same augmented reduction cell, interreduces the resulting Schreyer generators,
and returns an independently checked reduced basis in the exact induced source
frame. Before assembly, the general Schreyer colon-ideal criterion retains
only divisibility-minimal leading source multipliers for each basis position;
the complete unpruned schedule remains a test oracle on seeded module bases.
When every basis element is a single module monomial, those colon generators
give the exact two-term LCM relations directly. This theorem lane constructs
no F4 matrix, retains the same induced order, and records
`MonomialSchreyerTheorem` evidence instead of paying for a redundant
all-critical-pairs replay. Nonmonomial bases continue through the checked F4
lane below.
Independent unguided degree cells above the measured crossover are assigned by
a stable largest-work-first schedule to workspace-bounded private arenas and
merged back in ascending degree order. Hilbert-guided cells remain serial and
ascending so the exact early-stop theorem never speculates past its match.
Interreduction excludes the active element by index without cloning the rest
of the basis. Large final reducedness, generator-containment, and
critical-pair checks dispatch to deterministic Rayon chunks; every mutating
reduction owns a fork of the frozen monomial arena, and the workspace cap
bounds the worker count. Small checks and builds without the `parallel`
feature retain the serial correctness floor. Resolution iteration reuses that
certified basis directly instead of running a second generic module Gröbner
computation. Standard-graded
resolution steps may derive the exact next quotient Hilbert numerator from the
preceding exact sequence and stop after the first matching degree cell; every
returned relation still maps to zero and the resulting basis receives an
independent critical-pair sweep. Small pair sets stay on the unguided lane by a
measured crossover. Coefficient-generic monomial submodules use a direct monic
divisibility antichain while retaining exact coordinates in the original
generators.
Packed batch membership reuses one basis, monomial arena, and module order for
all queried values and exits at the first irreducible term of each nonmember;
Noether uses it for module containment and resolution image checks.
The tracked companion reduces a whole query family in that same packed frame
and retains exact ordered basis quotients. The original-generator kernel
operation combines those lifts with a checked basis change: if `B = F T` and
`F = B Q`, it transports Schreyer relations by `T` and adds the columns of
`I - T Q`. It verifies both displayed identities before returning portable
sparse relations. Prime-field module maps use this operation without first
projecting a public intermediate basis or change matrix.
Its generic exact-field reducer is the correctness floor; word-prime inputs
reuse the sparse/dense and `A|B / C|D` row dispatch. Noether independently
projects the verified basis and witness into its canonical free-module values.
Its coefficient-generic FGLM lane constructs sparse quotient multiplication
operators, performs bounded incremental order conversion, and verifies the
new basis together with both ideal containments. The same quotient engine owns
deterministic primitive-element search and exact rational univariate
representations, including source-equation substitution and square-free
reducedness checks.
Noether owns mathematical dispatch, canonical ideals, serialization, and
public errors.
